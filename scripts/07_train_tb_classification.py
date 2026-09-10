import sys
import os
import json
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class TBX11KDataset(Dataset):
    """PyTorch Dataset for TBX11K-Simplified dataset."""
    def __init__(self, df, img_dir, image_size=384, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['fname']
        img_path = self.img_dir / img_name
        
        # Binary target: 1 for TB, 0 for No TB
        label = 1.0 if row['target'] == 'tb' else 0.0

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img_tensor = torchvision.transforms.functional.to_tensor(img)
        img_tensor = torchvision.transforms.functional.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return img_tensor, torch.tensor(label, dtype=torch.float32)

def resolve_paths(data_csv_arg, img_dir_arg):
    candidate_csvs = [
        Path("data/raw/tuberculosis/data.csv"),
        Path(r"E:\kagglehub_cache\datasets\vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\data.csv"),
        Path(data_csv_arg)
    ]
    candidate_imgs = [
        Path("data/raw/tuberculosis/images"),
        Path(r"E:\kagglehub_cache\datasets\vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\images"),
        Path(img_dir_arg)
    ]

    csv_path = None
    for p in candidate_csvs:
        if p.exists():
            csv_path = p
            break
            
    img_path = None
    for p in candidate_imgs:
        if p.exists():
            img_path = p
            break

    return csv_path, img_path

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch GPU EfficientNetB0 on TBX11K Dataset.")
    parser.add_argument("--data-csv", type=str, 
                        default=r"data\raw\tuberculosis\data.csv",
                        help="Path to data.csv file.")
    parser.add_argument("--img-dir", type=str, 
                        default=r"data\raw\tuberculosis\images",
                        help="Path to images folder.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=384, help="Image resolution.")
    parser.add_argument("--subsample-ratio", type=float, default=1.0, help="Fraction of dataset to use (e.g., 0.2 for 20%).")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of training samples to use.")
    args = parser.parse_args()

    logger = setup_logger("train_tbx11k_torch")
    logger.info("=== STEP 06: PYTORCH GPU TBX11K TUBERCULOSIS CLASSIFICATION ===")

    csv_path, img_dir = resolve_paths(args.data_csv, args.img_dir)
    logger.info(f"Using Dataset CSV: {csv_path}")
    logger.info(f"Using Image Dir: {img_dir}")

    df = pd.read_csv(csv_path)
    train_df = df[df['source'] == 'train']
    val_df = df[df['source'] == 'val']

    # Apply Subsampling if requested
    if args.subsample_ratio < 1.0 or args.max_samples is not None:
        if args.subsample_ratio < 1.0:
            train_df = train_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
            val_df = val_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
        if args.max_samples is not None and len(train_df) > args.max_samples:
            train_df = train_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled Dataset Applied: Train={len(train_df)} | Val={len(val_df)}")

    logger.info(f"Loaded CSV: Total Samples={len(df)} | Train={len(train_df)} | Val={len(val_df)}")
    
    # Calculate positive weight for class imbalance
    num_neg = (train_df['target'] == 'no_tb').sum()
    num_pos = (train_df['target'] == 'tb').sum()
    pos_weight_val = num_neg / max(1, num_pos)
    logger.info(f"Class Imbalance in Train Set -> No TB (Neg): {num_neg} | TB (Pos): {num_pos} | pos_weight: {pos_weight_val:.2f}")

    train_dataset = TBX11KDataset(train_df, img_dir, image_size=args.image_size, is_train=True)
    val_dataset = TBX11KDataset(val_df, img_dir, image_size=args.image_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  - Active GPU Name: {torch.cuda.get_device_name(0)}")

    # Model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path("models/production")
    prod_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    total_steps = len(train_loader)
    logger.info(f"Starting PyTorch GPU Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | Steps per Epoch: {total_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(images).squeeze(-1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if batch_idx == 1 or batch_idx % max(1, total_steps // 5) == 0 or batch_idx == total_steps:
                pct = (batch_idx / total_steps) * 100
                acc = (correct / total) * 100
                logger.info(f"  [Epoch {epoch}/{args.epochs} | Step {batch_idx}/{total_steps} ({pct:.0f}%)] -> Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_tb_correct = 0
        val_tb_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze(-1)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                tb_mask = (labels == 1.0)
                if tb_mask.sum() > 0:
                    val_tb_correct += (preds[tb_mask] == 1.0).sum().item()
                    val_tb_total += tb_mask.sum().item()

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        val_tb_recall = (val_tb_correct / val_tb_total) * 100 if val_tb_total > 0 else 0.0
        elapsed = time.time() - start_time
        
        logger.info(f"=== EPOCH {epoch}/{args.epochs} COMPLETE in {elapsed:.1f}s | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}% | TB Sensitivity (Recall): {val_tb_recall:.2f}% ===")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = prod_dir / "best_tbx11k_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PyTorch TBX11K model to {save_path}")

    logger.info("=== PYTORCH GPU TBX11K TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()

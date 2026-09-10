import sys
import os
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class ImageClassificationFolderDataset(Dataset):
    """PyTorch Dataset for Folder-based Image Classification."""
    def __init__(self, folder_path, image_size=384):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        self.samples = []
        
        # Load 0 (Normal) and 1 (Pneumonia/Abnormal)
        for class_dir in sorted(self.folder_path.iterdir()):
            if class_dir.is_dir():
                label = 0 if class_dir.name == "0" or "normal" in class_dir.name.lower() else 1
                for img_p in class_dir.glob("*.*"):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".dcm"]:
                        self.samples.append((str(img_p), label))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img_tensor = torchvision.transforms.functional.to_tensor(img)
        # Normalize with ImageNet mean/std
        img_tensor = torchvision.transforms.functional.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return img_tensor, torch.tensor(label, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch GPU EfficientNetB0 Classification Model.")
    parser.add_argument("--data-dir", type=str, default="data/raw/dataset1", help="Path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=384, help="Image resolution.")
    parser.add_argument("--config", type=str, default=None, help="Config path.")
    args = parser.parse_args()

    logger = setup_logger("train_cls_torch")
    logger.info("=== STEP 05: PYTORCH GPU EFFICIENTNET CLASSIFICATION ===")

    config = load_config(args.config)
    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = ImageClassificationFolderDataset(train_dir, image_size=args.image_size)
    val_dataset = ImageClassificationFolderDataset(val_dir, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  - Active GPU Name: {torch.cuda.get_device_name(0)}")

    # Build EfficientNetB0
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path(config["paths"]["models_production"])
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

            if batch_idx == 1 or batch_idx % max(1, total_steps // 10) == 0 or batch_idx == total_steps:
                pct = (batch_idx / total_steps) * 100
                overall_pct = ((epoch - 1 + (batch_idx / total_steps)) / args.epochs) * 100
                acc = (correct / total) * 100
                logger.info(f"  [Epoch {epoch}/{args.epochs} | Step {batch_idx}/{total_steps} ({pct:.0f}%)] -> Total Progress: {overall_pct:.1f}% | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze(-1)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        elapsed = time.time() - start_time
        logger.info(f"=== EPOCH {epoch}/{args.epochs} COMPLETE in {elapsed:.1f}s | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}% ===")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = prod_dir / "best_classifier_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PyTorch GPU classification model to {save_path}")

    logger.info("=== PYTORCH GPU CLASSIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()

import sys
import os
import json
import time
import ast
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class TBX11KDetectionDataset(Dataset):
    """PyTorch Dataset for TBX11K Bounding Box Object Detection."""
    def __init__(self, df, img_dir, image_size=512):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['fname']
        img_path = self.img_dir / img_name
        orig_w = float(row.get('image_width', 512))
        orig_h = float(row.get('image_height', 512))

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img_tensor = torchvision.transforms.functional.to_tensor(img)

        boxes_scaled = []
        labels = []

        raw_bbox = row.get('bbox', 'none')
        if pd.notna(raw_bbox) and str(raw_bbox).strip().lower() != 'none':
            try:
                bdict = ast.literal_eval(str(raw_bbox))
                xmin = float(bdict['xmin'])
                ymin = float(bdict['ymin'])
                w = float(bdict['width'])
                h = float(bdict['height'])
                xmax = xmin + w
                ymax = ymin + h

                # Scale to resized target dimensions
                sx = self.image_size / orig_w
                sy = self.image_size / orig_h

                abs_xmin = max(0.0, xmin * sx)
                abs_ymin = max(0.0, ymin * sy)
                abs_xmax = min(float(self.image_size), xmax * sx)
                abs_ymax = min(float(self.image_size), ymax * sy)

                if abs_xmax > abs_xmin + 2.0 and abs_ymax > abs_ymin + 2.0:
                    boxes_scaled.append([abs_xmin, abs_ymin, abs_xmax, abs_ymax])
                    labels.append(1)  # Class 1 = Tuberculosis (Background is 0)
            except Exception:
                pass

        if len(boxes_scaled) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes_scaled, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

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
    parser = argparse.ArgumentParser(description="Train PyTorch RetinaNet Object Detector on TBX11K Dataset.")
    parser.add_argument("--data-csv", type=str, default=r"data\raw\tuberculosis\data.csv", help="Path to data.csv file.")
    parser.add_argument("--img-dir", type=str, default=r"data\raw\tuberculosis\images", help="Path to images folder.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for detection.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=384, help="Image resolution.")
    parser.add_argument("--subsample-ratio", type=float, default=1.0, help="Fraction of dataset to use (e.g., 0.2 for 20%).")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of training samples to use.")
    args = parser.parse_args()

    logger = setup_logger("train_tbx11k_detection_torch")
    logger.info("=== STEP 08: PYTORCH GPU TBX11K OBJECT DETECTION (RETINANET) ===")

    csv_path, img_dir = resolve_paths(args.data_csv, args.img_dir)
    logger.info(f"Using CSV: {csv_path}")
    logger.info(f"Using Image Dir: {img_dir}")

    df = pd.read_csv(csv_path)
    train_df = df[df['source'] == 'train'].reset_index(drop=True)
    val_df = df[df['source'] == 'val'].reset_index(drop=True)

    # Subsampling logic if requested
    if args.subsample_ratio < 1.0:
        train_df = train_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
    if args.max_samples is not None and len(train_df) > args.max_samples:
        train_df = train_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    logger.info(f"Loaded CSV: Train={len(train_df)} | Val={len(val_df)}")

    train_dataset = TBX11KDetectionDataset(train_df, img_dir, image_size=args.image_size)
    val_dataset = TBX11KDetectionDataset(val_df, img_dir, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  - Active GPU: {torch.cuda.get_device_name(0)}")

    # Pretrained RetinaNet
    model = retinanet_resnet50_fpn(weights=None, num_classes=2, weights_backbone=ResNet50_Weights.DEFAULT)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0001)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path("models/production")
    prod_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    total_steps = len(train_loader)
    logger.info(f"Starting GPU Object Detection Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | Image Size: {args.image_size} | Steps per Epoch: {total_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()

            if batch_idx == 1 or batch_idx % max(1, total_steps // 5) == 0 or batch_idx == total_steps:
                pct = (batch_idx / total_steps) * 100
                avg_loss = running_loss / batch_idx
                logger.info(f"  [Epoch {epoch}/{args.epochs} | Step {batch_idx}/{total_steps} ({pct:.0f}%)] -> Loss: {losses.item():.4f} (Avg: {avg_loss:.4f})")

        # Validation Loss
        model.train()  # Keep train mode to compute validation losses
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        elapsed = time.time() - start_time
        logger.info(f"=== EPOCH {epoch}/{args.epochs} COMPLETE in {elapsed:.1f}s | Train Loss: {running_loss/total_steps:.4f} | Val Loss: {avg_val_loss:.4f} ===")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = prod_dir / "best_tbx11k_detector_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PyTorch TB Detection model to {save_path}")

    logger.info("=== PYTORCH GPU TBX11K OBJECT DETECTION TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()

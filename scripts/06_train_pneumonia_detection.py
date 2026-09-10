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
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class ChestXRayDataset(Dataset):
    """PyTorch Dataset for Chest X-Ray Object Detection."""
    def __init__(self, records, image_size=512, transforms=None):
        self.records = records
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["image_path"]
        boxes = rec.get("boxes", [])
        cls_id = rec.get("class_id", 0) + 1  # Background is 0, Pneumonia is 1, TB is 2

        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            orig_w, orig_h = self.image_size, self.image_size

        # Convert image to Tensor (C, H, W) normalized [0, 1]
        img_tensor = torchvision.transforms.functional.to_tensor(img)

        # Scale bounding boxes to resized image coordinates [xmin, ymin, xmax, ymax]
        boxes_scaled = []
        labels = []

        for box in boxes:
            # box in json: [ymin, xmin, ymax, xmax] normalized 0..1
            ymin, xmin, ymax, xmax = box
            abs_xmin = max(0.0, xmin * self.image_size)
            abs_ymin = max(0.0, ymin * self.image_size)
            abs_xmax = min(float(self.image_size), xmax * self.image_size)
            abs_ymax = min(float(self.image_size), ymax * self.image_size)

            if abs_xmax > abs_xmin + 1.0 and abs_ymax > abs_ymin + 1.0:
                boxes_scaled.append([abs_xmin, abs_ymin, abs_xmax, abs_ymax])
                labels.append(cls_id)

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

def main():
    parser = argparse.ArgumentParser(description="Train PyTorch RetinaNet Chest X-Ray Detector on Compute Device.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader CPU worker processes (0 is required for Windows).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device: 'cuda' or 'cpu'.")
    parser.add_argument("--config", type=str, default=None, help="Config path.")
    args = parser.parse_args()

    logger = setup_logger("train_torch")
    logger.info("=== STEP 05: PYTORCH RETINANET TRAINING ===")

    config = load_config(args.config)
    paths = config["paths"]

    proc_dir = Path(paths["data_processed"])
    train_manifest = proc_dir / "train.json"
    val_manifest = proc_dir / "val.json"

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Target Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  - Active GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("  - Running on CPU.")

    image_size = config["model"].get("image_size", 512)
    train_dataset = ChestXRayDataset(train_records, image_size=image_size)
    val_dataset = ChestXRayDataset(val_records, image_size=image_size)

    pin_memory = (device.type == "cuda")
    logger.info(f"DataLoader Configuration: num_workers={args.num_workers} | pin_memory={pin_memory}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=pin_memory)

    # Initialize RetinaNet ResNet50 FPN model (3 classes: Background, Pneumonia, TB)
    model = retinanet_resnet50_fpn(weights=None, num_classes=3, weights_backbone=ResNet50_Weights.DEFAULT)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path(paths["models_production"])
    ckpt_dir = Path(paths["models_checkpoints"])
    prod_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    total_steps_per_epoch = len(train_loader)

    logger.info(f"Starting PyTorch Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | Steps per Epoch: {total_steps_per_epoch} | AMP: {device.type == 'cuda'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()

            # Log every 10% of epoch
            if batch_idx == 1 or batch_idx % max(1, total_steps_per_epoch // 10) == 0 or batch_idx == total_steps_per_epoch:
                pct = (batch_idx / total_steps_per_epoch) * 100
                overall_pct = ((epoch - 1 + (batch_idx / total_steps_per_epoch)) / args.epochs) * 100
                logger.info(f"  [Epoch {epoch}/{args.epochs} | Step {batch_idx}/{total_steps_per_epoch} ({pct:.0f}%)] -> Total Progress: {overall_pct:.1f}% | Loss: {losses.item():.4f}")

        epoch_loss = running_loss / total_steps_per_epoch
        elapsed = time.time() - start_time
        logger.info(f"=== EPOCH {epoch}/{args.epochs} FINISHED in {elapsed:.1f}s | Train Loss: {epoch_loss:.4f} ===")

        # Save best model checkpoint
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            save_path = prod_dir / "best_model_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best PyTorch GPU model to {save_path}")

    logger.info("=== PYTORCH GPU TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()

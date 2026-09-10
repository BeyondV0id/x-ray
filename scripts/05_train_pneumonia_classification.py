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
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class ImageClassificationFolderDataset(Dataset):
    """PyTorch Dataset for Folder-based Image Classification with proper augmentation."""

    def __init__(self, folder_path, image_size=384, is_train=True):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        self.is_train = is_train
        self.samples = []

        for class_dir in sorted(self.folder_path.iterdir()):
            if class_dir.is_dir():
                label = 0 if class_dir.name == "0" or "normal" in class_dir.name.lower() else 1
                for img_p in class_dir.glob("*.*"):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".dcm"]:
                        self.samples.append((str(img_p), label))

        # Build transforms
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # RandomErasing: suppresses text labels, border markers, and DICOM annotations
                # that the model would otherwise learn as shortcuts
                T.RandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# Label Smoothing BCE Loss
# ──────────────────────────────────────────────────────────────────────
class LabelSmoothingBCELoss(nn.Module):
    """BCE with label smoothing to prevent overconfident shortcut predictions."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth labels: 0 → smoothing/2, 1 → 1 - smoothing/2
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


# ──────────────────────────────────────────────────────────────────────
# Backbone freeze/unfreeze helpers
# ──────────────────────────────────────────────────────────────────────
def freeze_backbone(model):
    """Freeze all EfficientNet feature layers; only classifier is trainable."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_deep_layers(model, from_block: int = 5):
    """Unfreeze EfficientNet features from `from_block` onwards.

    EfficientNetB0 features layout:
      0: stem Conv2dNormActivation
      1: MBConv1 block
      2: MBConv6 block (stride 2)
      3: MBConv6 block (stride 2)
      4: MBConv6 block (stride 2)
      5: MBConv6 block (stride 1)  ← semantic-rich, spatial still good
      6: MBConv6 block (stride 2)  ← best Grad-CAM layer
      7: MBConv6 block (stride 1)
      8: top Conv2dNormActivation
    """
    for i, block in enumerate(model.features):
        if i >= from_block:
            for param in block.parameters():
                param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train PyTorch EfficientNetB0 Pneumonia Classifier (v2 — fixed augmentation + staged unfreezing).")
    parser.add_argument("--data-dir", type=str, default="data/raw/dataset1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Epochs to train with frozen backbone before staged unfreezing.")
    parser.add_argument("--unfreeze-from", type=int, default=5,
                        help="EfficientNet features block index to start unfreezing from.")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger("train_cls_torch_v2")
    logger.info("=== STEP 05 v2: EFFICIENTNET PNEUMONIA CLASSIFICATION (Fixed Training) ===")

    config = load_config(args.config)
    data_dir = Path(args.data_dir).resolve()
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = ImageClassificationFolderDataset(train_dir, image_size=args.image_size, is_train=True)
    val_dataset   = ImageClassificationFolderDataset(val_dir,   image_size=args.image_size, is_train=False)

    logger.info(f"Dataset — Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ──────────────────────────────────────────────────────────
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    # Phase 1: freeze backbone — only classifier trains
    freeze_backbone(model)
    logger.info(f"Phase 1 ({args.warmup_epochs} epochs): Backbone FROZEN — training classifier head only.")

    criterion = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path(config["paths"]["models_production"])
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing model before overwriting
    existing = prod_dir / "best_classifier_pytorch.pth"
    if existing.exists():
        backup = prod_dir / "best_classifier_pytorch_backup.pth"
        shutil.copy2(existing, backup)
        logger.info(f"Backed up existing model → {backup}")

    best_val_auc = 0.0
    total_steps = len(train_loader)

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2: unfreeze deep blocks after warmup ──────────────────
        if epoch == args.warmup_epochs + 1:
            unfreeze_deep_layers(model, from_block=args.unfreeze_from)
            # Rebuild optimizer with all trainable params and reduced LR for backbone
            optimizer = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": args.lr},
                {"params": [p for n, p in model.features.named_parameters() if p.requires_grad],
                 "lr": args.lr * 0.1},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
            logger.info(f"Phase 2 (epoch {epoch}+): Unfreezing features[{args.unfreeze_from}:] with LR×0.1 for backbone.")

        # ── Train ───────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
                acc = (correct / total) * 100
                overall_pct = ((epoch - 1 + batch_idx / total_steps) / args.epochs) * 100
                logger.info(
                    f"  [Epoch {epoch}/{args.epochs} | Step {batch_idx}/{total_steps}] "
                    f"Total: {overall_pct:.1f}% | Loss: {loss.item():.4f} | Acc: {acc:.2f}%"
                )

        scheduler.step()
        train_acc = (correct / total) * 100

        # ── Validate ────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images).squeeze(-1)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0
        try:
            val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        except Exception:
            val_auc = 0.0

        elapsed = time.time() - start_time
        logger.info(
            f"=== EPOCH {epoch}/{args.epochs} COMPLETE in {elapsed:.1f}s | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"Val AUC: {val_auc:.4f} | LR: {scheduler.get_last_lr()} ==="
        )

        # Save checkpoint on best Val AUC (not accuracy)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = prod_dir / "best_classifier_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ New best Val AUC={val_auc:.4f} — saved to {save_path}")

    logger.info(f"=== TRAINING COMPLETE | Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    main()

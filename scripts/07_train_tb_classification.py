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
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class TBX11KDataset(Dataset):
    """PyTorch Dataset for TBX11K with proper medical imaging augmentation."""

    def __init__(self, df, img_dir, image_size=384, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.is_train = is_train

        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Erase small patches to prevent relying on border text / annotations
                T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['fname']
        label = 1.0 if row['target'] == 'tb' else 0.0
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# Label Smoothing BCE Loss
# ──────────────────────────────────────────────────────────────────────
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """Weighted BCE + label smoothing to curb shortcut-feature overconfidence."""
    def __init__(self, pos_weight: torch.Tensor = None, smoothing: float = 0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight
        )


# ──────────────────────────────────────────────────────────────────────
# Backbone helpers
# ──────────────────────────────────────────────────────────────────────
def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_deep_layers(model, from_block: int = 5):
    """Unfreeze EfficientNetB0 feature blocks from `from_block` onwards.

    Block layout (EfficientNetB0):
      0 - Stem Conv
      1 - MBConv1
      2 - MBConv6 ×2
      3 - MBConv6 ×2
      4 - MBConv6 ×3
      5 - MBConv6 ×3   ← semantics + decent spatial resolution
      6 - MBConv6 ×4   ← BEST Grad-CAM target (24×24 @ 384px)
      7 - MBConv6 ×1
      8 - Top Conv
    """
    for i, block in enumerate(model.features):
        if i >= from_block:
            for param in block.parameters():
                param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────
# Path resolver
# ──────────────────────────────────────────────────────────────────────
def resolve_paths(data_csv_arg, img_dir_arg):
    candidate_csvs = [
        Path("data/raw/tuberculosis/data.csv"),
        Path(r"E:\kagglehub_cache\datasets\vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\data.csv"),
        Path(data_csv_arg),
    ]
    candidate_imgs = [
        Path("data/raw/tuberculosis/images"),
        Path(r"E:\kagglehub_cache\datasets\vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\images"),
        Path(img_dir_arg),
    ]
    csv_path = next((p for p in candidate_csvs if p.exists()), None)
    img_path = next((p for p in candidate_imgs if p.exists()), None)
    return csv_path, img_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 on TBX11K (v2 — fixed augmentation + staged unfreeze + AUC checkpoint)."
    )
    parser.add_argument("--data-csv",  type=str, default=r"data\raw\tuberculosis\data.csv")
    parser.add_argument("--img-dir",   type=str, default=r"data\raw\tuberculosis\images")
    parser.add_argument("--epochs",    type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",        type=float, default=0.0001)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--subsample-ratio", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Epochs with frozen backbone before staged unfreezing.")
    parser.add_argument("--unfreeze-from", type=int, default=5,
                        help="Feature block index to start unfreezing from.")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger("train_tbx11k_torch_v2")
    logger.info("=== STEP 07 v2: EFFICIENTNET TB CLASSIFICATION (Fixed Training) ===")

    csv_path, img_dir = resolve_paths(args.data_csv, args.img_dir)
    logger.info(f"Dataset CSV : {csv_path}")
    logger.info(f"Images dir  : {img_dir}")

    df = pd.read_csv(csv_path)
    train_df = df[df['source'] == 'train'].copy()
    val_df   = df[df['source'] == 'val'].copy()

    if args.subsample_ratio < 1.0 or args.max_samples is not None:
        if args.subsample_ratio < 1.0:
            train_df = train_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
            val_df   = val_df.sample(frac=args.subsample_ratio, random_state=42).reset_index(drop=True)
        if args.max_samples and len(train_df) > args.max_samples:
            train_df = train_df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled: Train={len(train_df)} | Val={len(val_df)}")

    num_neg = (train_df['target'] == 'no_tb').sum()
    num_pos = (train_df['target'] == 'tb').sum()
    pos_weight_val = num_neg / max(1, num_pos)
    logger.info(f"Class balance — Neg: {num_neg} | Pos: {num_pos} | pos_weight: {pos_weight_val:.2f}")
    logger.info(f"Train total: {len(train_df)} | Val total: {len(val_df)}")

    train_dataset = TBX11KDataset(train_df, img_dir, image_size=args.image_size, is_train=True)
    val_dataset   = TBX11KDataset(val_df,   img_dir, image_size=args.image_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ──────────────────────────────────────────────────────────
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.to(device)

    # Freeze backbone for warmup
    freeze_backbone(model)
    logger.info(f"Phase 1 ({args.warmup_epochs} warmup epochs): Backbone FROZEN.")

    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = LabelSmoothingBCEWithLogitsLoss(pos_weight=pos_weight_tensor, smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    prod_dir = Path("models/production")
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing model
    existing = prod_dir / "best_tbx11k_pytorch.pth"
    if existing.exists():
        backup = prod_dir / "best_tbx11k_pytorch_backup.pth"
        shutil.copy2(existing, backup)
        logger.info(f"Backed up existing TB model → {backup}")

    best_val_auc = 0.0
    total_steps  = len(train_loader)

    for epoch in range(1, args.epochs + 1):

        # ── Phase 2 unfreeze ────────────────────────────────────────────
        if epoch == args.warmup_epochs + 1:
            unfreeze_deep_layers(model, from_block=args.unfreeze_from)
            optimizer = torch.optim.AdamW([
                {"params": model.classifier.parameters(), "lr": args.lr},
                {"params": [p for n, p in model.features.named_parameters() if p.requires_grad],
                 "lr": args.lr * 0.1},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7
            )
            logger.info(f"Phase 2 (epoch {epoch}+): Unfreezing features[{args.unfreeze_from}:] with LR×0.1.")

        # ── Train ───────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
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
            preds   = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

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
        val_correct  = 0
        val_total    = 0
        val_tb_correct = 0
        val_tb_total   = 0
        all_labels   = []
        all_probs    = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images).squeeze(-1)
                probs   = torch.sigmoid(outputs)
                preds   = (probs >= 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

                tb_mask = (labels == 1.0)
                if tb_mask.sum() > 0:
                    val_tb_correct += (preds[tb_mask] == 1.0).sum().item()
                    val_tb_total   += tb_mask.sum().item()

        val_acc      = (val_correct    / val_total)    * 100 if val_total    > 0 else 0.0
        val_tb_recall = (val_tb_correct / val_tb_total) * 100 if val_tb_total > 0 else 0.0
        try:
            val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        except Exception:
            val_auc = 0.0

        elapsed = time.time() - start_time
        logger.info(
            f"=== EPOCH {epoch}/{args.epochs} in {elapsed:.1f}s | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"TB Sensitivity: {val_tb_recall:.2f}% | Val AUC: {val_auc:.4f} ==="
        )

        # Checkpoint on best Val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = prod_dir / "best_tbx11k_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ New best Val AUC={val_auc:.4f} — saved to {save_path}")

    logger.info(f"=== TB TRAINING COMPLETE | Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    main()

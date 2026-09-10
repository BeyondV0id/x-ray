"""
Script 13: Evaluate PyTorch GPU Classification Model on Dataset1 (Pneumonia)
=============================================================================
Evaluates the trained EfficientNetB0 classification model on Dataset 1
(data/raw/dataset1/val or data/raw/dataset1/test).

Computes Accuracy, Precision, Recall (Sensitivity), F1-Score, Specificity, 
Confusion Matrix, and saves diagnostic plots.

Usage:
    python scripts/13_test_classification_dataset1.py
    python scripts/13_test_classification_dataset1.py --split test
    python scripts/13_test_classification_dataset1.py --batch-size 16
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class Dataset1ClassificationDataset(Dataset):
    """PyTorch Dataset for Dataset1 Folder Structure (0 = Normal, 1 = Pneumonia)."""
    def __init__(self, folder_path, image_size=384):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        self.samples = []
        
        # Folder 0: Normal, Folder 1: Pneumonia
        for class_dir in sorted(self.folder_path.iterdir()):
            if class_dir.is_dir():
                label = 0.0 if class_dir.name == "0" or "normal" in class_dir.name.lower() else 1.0
                for img_p in sorted(class_dir.glob("*.*")):
                    if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".dcm"]:
                        self.samples.append((str(img_p), label, img_p.name))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, fname = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, torch.tensor(label, dtype=torch.float32), fname

def load_classification_model(model_path: Path, device: torch.device):
    """Loads the trained EfficientNetB0 classification checkpoint."""
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Classification Model on Dataset1 (Pneumonia).")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Split to evaluate (default: val).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--image-size", type=int, default=384, help="Image resolution (default: 384).")
    parser.add_argument("--model-path", type=str, default="models/production/best_classifier_pytorch.pth", help="Model checkpoint path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold (default: 0.5).")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate (default: all).")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logger("test_cls_dataset1")
    logger.info("=" * 60)
    logger.info(f"  DATASET1 CLASSIFICATION VALIDATION  (split={args.split})")
    logger.info("=" * 60)

    config = load_config(args.config)
    paths = config["paths"]
    
    dataset1_dir = Path(paths["data_raw"]) / "dataset1" / args.split
    model_path = Path(args.model_path)

    if not model_path.exists():
        # Fallback to best_tbx11k_pytorch.pth if best_classifier_pytorch.pth is missing
        fallback = Path("models/production/best_tbx11k_pytorch.pth")
        if fallback.exists():
            logger.info(f"Primary model not found, using fallback checkpoint: {fallback}")
            model_path = fallback
        else:
            logger.error(f"Classification model checkpoint not found at {model_path}")
            sys.exit(1)

    if not dataset1_dir.exists():
        logger.error(f"Dataset 1 split directory not found at {dataset1_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")

    dataset = Dataset1ClassificationDataset(dataset1_dir, image_size=args.image_size)
    if args.max_samples and args.max_samples < len(dataset):
        norm_samples = [s for s in dataset.samples if s[1] == 0.0]
        pos_samples  = [s for s in dataset.samples if s[1] == 1.0]
        half = args.max_samples // 2
        dataset.samples = norm_samples[:half] + pos_samples[:half]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Loaded {len(dataset)} samples from {dataset1_dir}")

    model = load_classification_model(model_path, device)
    logger.info(f"Loaded classification model from {model_path}")

    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, labels, fnames in loader:
            images = images.to(device)
            outputs = model(images).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_true.extend(labels.numpy())
            y_probs.extend(probs)

    y_true = np.array(y_true, dtype=np.int32)
    y_probs = np.array(y_probs, dtype=np.float32)
    y_preds = (y_probs >= args.threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = tn / max(1, (tn + fp))

    print("\n" + "=" * 60)
    print(f"   DATASET 1 CLASSIFICATION RESULTS — {args.split.upper()} SPLIT")
    print("=" * 60)
    print(f"  Total Evaluated Samples : {len(y_true)}")
    print(f"  Normal (Class 0)        : {tn + fp}")
    print(f"  Pneumonia (Class 1)     : {tp + fn}")
    print("-" * 60)
    print(f"  True  Positives (TP)    : {tp}")
    print(f"  False Positives (FP)    : {fp}")
    print(f"  False Negatives (FN)    : {fn}")
    print(f"  True  Negatives (TN)    : {tn}")
    print("-" * 60)
    print(f"  Accuracy                : {acc * 100:.2f}%")
    print(f"  Sensitivity (Recall)    : {rec * 100:.2f}%")
    print(f"  Specificity             : {spec * 100:.2f}%")
    print(f"  Precision               : {prec * 100:.2f}%")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"  ROC-AUC Score           : {auc:.4f}")
    print("=" * 60 + "\n")

    # Plot Confusion Matrix and ROC Curve
    plots_dir = Path(paths["results_plots"])
    metrics_dir = Path(paths["results_metrics"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title(f"Dataset 1 Confusion Matrix ({args.split.upper()})", fontsize=12, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, color="#2b6cb0", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    axes[1].fill_between(fpr, tpr, alpha=0.15, color="#2b6cb0")
    axes[1].plot([0, 1], [0, 1], color="#e53e3e", lw=1.5, linestyle="--")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    axes[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    axes[1].set_title(f"Dataset 1 ROC Curve ({args.split.upper()})", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plot_path = plots_dir / f"dataset1_classification_{args.split}_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    metrics_out = {
        "split": args.split,
        "n_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    }
    json_path = metrics_dir / f"dataset1_classification_{args.split}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info(f"Saved diagnostic plots to: {plot_path}")
    logger.info(f"Saved evaluation metrics JSON to: {json_path}")
    logger.info("=== DONE ===")

if __name__ == "__main__":
    main()

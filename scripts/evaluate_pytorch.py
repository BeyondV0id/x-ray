import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

class SimpleImageEvalDataset(torch.utils.data.Dataset):
    def __init__(self, samples, image_size=384):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # Normalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, torch.tensor(label, dtype=torch.float32)

def evaluate_tb_model(model_path: Path, csv_path: Path, img_dir: Path, output_metrics_dir: Path, output_plots_dir: Path, image_size=384):
    logger = setup_logger("eval_pytorch")
    logger.info("=== EVALUATING PYTORCH TUBERCULOSIS (TBX11K) MODEL ON TEST SET ===")

    df = pd.read_csv(csv_path)
    val_df = df[df['source'] == 'val'].reset_index(drop=True)
    
    samples = []
    for idx, row in val_df.iterrows():
        img_p = img_dir / row['fname']
        label = 1.0 if row['target'] == 'tb' else 0.0
        if img_p.exists():
            samples.append((str(img_p), label))

    logger.info(f"Loaded {len(samples)} validation test samples.")
    dataset = SimpleImageEvalDataset(samples, image_size=image_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device}")

    # Build model
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred_probs.extend(probs)

    y_true = np.array(y_true, dtype=np.int32)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    y_pred_labels = (y_pred_probs >= 0.5).astype(np.int32)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels, zero_division=0)
    rec = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "="*60)
    print("      📊 STATISTICAL EVALUATION REPORT (TB MODEL)")
    print("="*60)
    print(f"  • Overall Accuracy:    {acc*100:.2f}%  (Overall correctness)")
    print(f"  • ROC AUC Score:       {auc:.4f}     (Area Under Curve)")
    print(f"  • Sensitivity/Recall:  {rec*100:.2f}%  (TB Detection Rate)")
    print(f"  • Specificity:         {specificity*100:.2f}%  (Normal Detection Rate)")
    print(f"  • Precision:           {prec*100:.2f}%  (Positive Predictive Value)")
    print(f"  • F1-Score:            {f1:.4f}     (Harmonic Mean)")
    print(f"  • Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*60 + "\n")

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No TB", "TB"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Tuberculosis (TBX11K) Confusion Matrix")
    cm_path = output_plots_dir / "tb_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkred", lw=2, label=f"TB ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Tuberculosis ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    roc_path = output_plots_dir / "tb_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Metrics JSON
    metrics_dict = {
        "model": "EfficientNetB0_TBX11K",
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": float(prec),
        "sensitivity_recall": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    json_path = output_metrics_dir / "tb_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Evaluation completed. Metrics & plots saved to:")
    logger.info(f"  • {json_path}")
    logger.info(f"  • {cm_path}")
    logger.info(f"  • {roc_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch GPU Models.")
    parser.add_argument("--model-path", type=str, default="models/production/best_tbx11k_pytorch.pth")
    parser.add_argument("--data-csv", type=str, default=r"data_zips\tbx11k-simplified\data.csv")
    parser.add_argument("--img-dir", type=str, default=r"data_zips\tbx11k-simplified\images")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    
    metrics_dir = Path(paths["results_metrics"])
    plots_dir = Path(paths["results_plots"])

    evaluate_tb_model(
        model_path=Path(args.model_path),
        csv_path=Path(args.data_csv),
        img_dir=Path(args.img_dir),
        output_metrics_dir=metrics_dir,
        output_plots_dir=plots_dir
    )

if __name__ == "__main__":
    main()

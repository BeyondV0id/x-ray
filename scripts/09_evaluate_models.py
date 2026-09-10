import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
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
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, torch.tensor(label, dtype=torch.float32)

def generate_learning_curves(history_json_path: Path, output_plots_dir: Path):
    """Generates Training & Validation Accuracy and Loss Curves vs Epochs."""
    logger = setup_logger("learning_curves")
    if not history_json_path.exists():
        logger.warning(f"Training history JSON not found at {history_json_path}. Skipping epoch learning curves.")
        return

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("accuracy", [])) + 1)
    if not epochs:
        return

    output_plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy vs Epochs Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [a * 100 if a <= 1.0 else a for a in history["accuracy"]], 'o-', color='#2b6cb0', lw=2.5, label='Training Accuracy')
    ax.plot(epochs, [a * 100 if a <= 1.0 else a for a in history["val_accuracy"]], 's--', color='#38a169', lw=2.5, label='Validation Accuracy')
    ax.set_title("Training & Validation Accuracy vs Epochs (Overfitting Check)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch Number", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    acc_path = output_plots_dir / "learning_curve_accuracy.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # 2. Loss vs Epochs Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["loss"], 'o-', color='#e53e3e', lw=2.5, label='Training Loss')
    ax.plot(epochs, history["val_loss"], 's--', color='#dd6b20', lw=2.5, label='Validation Loss')
    ax.set_title("Training & Validation Loss vs Epochs (Convergence Check)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch Number", fontsize=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    loss_path = output_plots_dir / "learning_curve_loss.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    logger.info(f"Generated Epoch Learning Curves:\n  - {acc_path}\n  - {loss_path}")

def generate_classwise_comparison_chart(tn, fp, fn, tp, output_plots_dir: Path):
    """Generates Class-wise Precision, Recall, and F1-score comparison for Normal vs Abnormal."""
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    # Class 0: Normal
    prec_norm = tn / max(1, (tn + fn))
    rec_norm = tn / max(1, (tn + fp))
    f1_norm = 2 * (prec_norm * rec_norm) / max(1e-6, (prec_norm + rec_norm))

    # Class 1: Abnormal / Positive
    prec_abnorm = tp / max(1, (tp + fp))
    rec_abnorm = tp / max(1, (tp + fn))
    f1_abnorm = 2 * (prec_abnorm * rec_abnorm) / max(1e-6, (prec_abnorm + rec_abnorm))

    labels = ['Precision', 'Recall (Sensitivity)', 'F1-Score']
    normal_scores = [prec_norm * 100, rec_norm * 100, f1_norm * 100]
    abnormal_scores = [prec_abnorm * 100, rec_abnorm * 100, f1_abnorm * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, normal_scores, width, label='Normal (Class 0)', color='#3182ce')
    rects2 = ax.bar(x + width/2, abnormal_scores, width, label='Abnormal / Disease (Class 1)', color='#e53e3e')

    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title('Class-wise Performance Comparison (Normal vs Abnormal)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")
    ax.grid(axis='y', linestyle=":", alpha=0.6)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    chart_path = output_plots_dir / "classwise_performance_comparison.png"
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("      CLASS-WISE METRICS BREAKDOWN (NORMAL VS ABNORMAL)")
    print("="*60)
    print(f"  • Normal (Class 0):   Precision={prec_norm*100:.2f}% | Recall={rec_norm*100:.2f}% | F1={f1_norm:.4f}")
    print(f"  • Abnormal (Class 1): Precision={prec_abnorm*100:.2f}% | Recall={rec_abnorm*100:.2f}% | F1={f1_abnorm:.4f}")
    print("="*60 + "\n")

def evaluate_tb_model(model_path: Path, csv_path: Path, img_dir: Path, output_metrics_dir: Path, output_plots_dir: Path, image_size=384):
    logger = setup_logger("eval_pytorch")
    logger.info("=== EVALUATING PYTORCH MODEL & GENERATING COMPLETE DIAGNOSTIC PLOTS ===")

    df = pd.read_csv(csv_path)
    val_df = df[df['source'] == 'val'].reset_index(drop=True)
    
    samples = []
    for idx, row in val_df.iterrows():
        img_p = img_dir / row['fname']
        label = 1.0 if row['target'] == 'tb' else 0.0
        if img_p.exists():
            samples.append((str(img_p), label))

    dataset = SimpleImageEvalDataset(samples, image_size=image_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ap = average_precision_score(y_true, y_pred_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal / Disease"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Normal vs Abnormal)", fontsize=12, fontweight='bold')
    cm_path = output_plots_dir / "tb_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Shaded ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#d9534f", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#d9534f")
    ax.plot([0, 1], [0, 1], color="#2b6cb0", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC Curve)", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    roc_path = output_plots_dir / "tb_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curve Plot
    p_vals, r_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_vals, p_vals, color="#20c997", lw=2.5, label=f"Precision-Recall Curve (AP = {ap:.4f})")
    ax.fill_between(r_vals, p_vals, alpha=0.15, color="#20c997")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.set_title("Precision-Recall Curve (Imbalanced Dataset Evaluation)", fontsize=12, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    pr_path = output_plots_dir / "tb_precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # 6. Class-wise Comparison Chart
    generate_classwise_comparison_chart(tn, fp, fn, tp, output_plots_dir)

def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch GPU Models & Generate Complete Diagnostic Plots.")
    parser.add_argument("--tb-model-path", type=str, default="models/production/best_tbx11k_pytorch.pth")
    parser.add_argument("--det-model-path", type=str, default="models/production/best_model_pytorch.pth")
    parser.add_argument("--history-path", type=str, default="models/production/training_history_classification.json")
    parser.add_argument("--data-csv", type=str, default=r"data_zips\tbx11k-simplified\data.csv")
    parser.add_argument("--img-dir", type=str, default=r"data_zips\tbx11k-simplified\images")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    
    metrics_dir = Path(paths["results_metrics"])
    plots_dir = Path(paths["results_plots"])

    # 4 & 5. Generate Accuracy & Loss Learning Curves vs Epochs
    history_json = Path(args.history_path)
    generate_learning_curves(history_json, plots_dir)

    # 1, 2, 3, 6. Generate Confusion Matrix, ROC, PR-Curve, Classwise Comparison
    tb_ckpt = Path(args.tb_model_path)
    if tb_ckpt.exists():
        evaluate_tb_model(
            model_path=tb_ckpt,
            csv_path=Path(args.data_csv),
            img_dir=Path(args.img_dir),
            output_metrics_dir=metrics_dir,
            output_plots_dir=plots_dir
        )

if __name__ == "__main__":
    main()

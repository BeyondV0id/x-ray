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

def compute_box_iou(box1, box2):
    """Compute Intersection-over-Union (IoU) of two bounding boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def evaluate_detection_model(model_path: Path, val_json_path: Path, output_metrics_dir: Path, output_plots_dir: Path, score_thresh=0.2, iou_thresh=0.5, image_size=512):
    """Evaluates PyTorch RetinaNet Object Detector on validation manifest with ground truth bounding boxes."""
    logger = setup_logger("eval_detection")
    logger.info("=== EVALUATING PYTORCH RETINANET OBJECT DETECTION MODEL ===")

    if not model_path.exists():
        logger.warning(f"Detection model checkpoint not found at {model_path}. Skipping detection evaluation.")
        return

    if not val_json_path.exists():
        logger.warning(f"Validation JSON manifest not found at {val_json_path}. Skipping detection evaluation.")
        return

    with open(val_json_path, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device}")

    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import retinanet_resnet50_fpn

    model = retinanet_resnet50_fpn(weights=None, num_classes=3, weights_backbone=ResNet50_Weights.DEFAULT)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_gt_boxes = 0
    total_pred_boxes = 0
    tp_boxes = 0
    fp_boxes = 0
    fn_boxes = 0
    iou_scores = []

    for rec in val_records:
        img_path = rec["image_path"]
        gt_boxes = rec.get("boxes", [])

        abs_gt_boxes = []
        for box in gt_boxes:
            ymin, xmin, ymax, xmax = box
            abs_gt_boxes.append([xmin * image_size, ymin * image_size, xmax * image_size, ymax * image_size])
        total_gt_boxes += len(abs_gt_boxes)

        try:
            img = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        except Exception:
            continue

        with torch.no_grad():
            outputs = model([img_tensor.to(device)])[0]

        pred_boxes = outputs["boxes"].cpu().numpy()
        pred_scores = outputs["scores"].cpu().numpy()
        pred_labels = outputs["labels"].cpu().numpy()

        keep = pred_scores >= score_thresh
        keep_boxes = pred_boxes[keep]
        total_pred_boxes += len(keep_boxes)

        gt_matched = [False] * len(abs_gt_boxes)
        for p_box in keep_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g_box in enumerate(abs_gt_boxes):
                iou = compute_box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                tp_boxes += 1
                gt_matched[best_gt_idx] = True
                iou_scores.append(float(best_iou))
            else:
                fp_boxes += 1

        fn_boxes += sum(1 for matched in gt_matched if not matched)

    precision = tp_boxes / max(1, (tp_boxes + fp_boxes))
    recall = tp_boxes / max(1, (tp_boxes + fn_boxes))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    metrics = {
        "score_threshold": score_thresh,
        "iou_threshold": iou_thresh,
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "true_positives": tp_boxes,
        "false_positives": fp_boxes,
        "false_negatives": fn_boxes,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4)
    }

    out_metrics_path = output_metrics_dir / "detection_evaluation_metrics.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("      OBJECT DETECTION VALIDATION METRICS (RETINANET)")
    print("="*60)
    print(f"  • Total GT Bounding Boxes  : {total_gt_boxes}")
    print(f"  • Total Predicted Boxes    : {total_pred_boxes}")
    print(f"  • Bounding Box Precision   : {precision * 100:.2f}%")
    print(f"  • Bounding Box Recall      : {recall * 100:.2f}%")
    print(f"  • Bounding Box F1-Score    : {f1:.4f}")
    print(f"  • Mean Bounding Box IoU    : {mean_iou:.4f}")
    print("="*60 + "\n")

    logger.info(f"Saved Detection Metrics to {out_metrics_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch GPU Models & Generate Complete Diagnostic Plots.")
    parser.add_argument("--tb-model-path", type=str, default="models/production/best_tbx11k_pytorch.pth")
    parser.add_argument("--det-model-path", type=str, default="models/production/best_model_pytorch.pth")
    parser.add_argument("--val-json-path", type=str, default="data/processed/val.json")
    parser.add_argument("--history-path", type=str, default="models/production/training_history_classification.json")
    parser.add_argument("--data-csv", type=str, default=r"data\raw\tuberculosis\data.csv")
    parser.add_argument("--img-dir", type=str, default=r"data\raw\tuberculosis\images")
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

    # 7. Evaluate Object Detection Model on Validation JSON Manifest
    det_ckpt = Path(args.det_model_path)
    val_manifest = Path(args.val_json_path)
    if det_ckpt.exists() and val_manifest.exists():
        evaluate_detection_model(
            model_path=det_ckpt,
            val_json_path=val_manifest,
            output_metrics_dir=metrics_dir,
            output_plots_dir=plots_dir
        )

if __name__ == "__main__":
    main()

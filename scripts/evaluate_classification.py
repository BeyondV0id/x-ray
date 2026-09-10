import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Keras Classification Model on Test Dataset.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model (.keras file).")
    parser.add_argument("--data-dir", type=str, default="data/raw/dataset1", help="Path to dataset root directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("evaluate_classification")
    logger.info("=== EVALUATING CLASSIFICATION MODEL ON TEST DATASET ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    paths = config["paths"]
    data_dir = Path(args.data_dir).resolve()
    test_dir = data_dir / "test"

    if not test_dir.exists():
        logger.error(f"Test directory not found at {test_dir}.")
        sys.exit(1)

    model_path = Path(args.model_path) if args.model_path else Path(paths["models_production"]) / "best_model.keras"

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}.")
        sys.exit(1)

    logger.info(f"Loading trained classification model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    image_size = args.image_size if hasattr(args, "image_size") else config["model"].get("image_size", 512)

    logger.info(f"Loading test dataset from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=32,
        shuffle=False,
        label_mode="binary"
    )

    # Extract ground truth labels and predicted probabilities
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    y_pred_labels = (y_pred_probs >= 0.5).astype(np.int32)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels)
    rec = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc = roc_auc_score(y_true, y_pred_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("\n" + "="*50)
    logger.info("=== CLASSIFICATION EVALUATION METRICS ===")
    logger.info(f"  - Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  - ROC AUC:     {auc:.4f}")
    logger.info(f"  - Precision:   {prec:.4f}")
    logger.info(f"  - Sensitivity (Recall): {rec:.4f}")
    logger.info(f"  - Specificity: {specificity:.4f}")
    logger.info(f"  - F1-Score:    {f1:.4f}")
    logger.info(f"  - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info("="*50 + "\n")

    results_dir = Path(paths["results_metrics"])
    plots_dir = Path(paths["results_plots"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Classification Confusion Matrix")
    cm_path = plots_dir / "classification_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    roc_path = plots_dir / "classification_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Save Summary JSON
    report_dict = {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    json_path = results_dir / "classification_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Evaluation results and plots saved to:\n  - {cm_path}\n  - {roc_path}\n  - {json_path}")
    logger.info("=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed
from src.dataset import load_image_as_rgb
from src.model import build_retinanet_model, decode_predictions, FocalLoss, SmoothL1Loss
from src.metrics import calculate_map_metrics
from src.visualization import draw_detections

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RetinaNet model on test dataset.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model (.keras file).")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("evaluate")
    logger.info("=== STEP 06: MODEL EVALUATION ON TEST DATASET ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    paths = config["paths"]
    test_manifest = Path(paths["data_processed"]) / "test.json"

    if not test_manifest.exists():
        logger.error(f"Test manifest missing at {test_manifest}. Run 03_prepare_dataset.py first.")
        sys.exit(1)

    with open(test_manifest, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    logger.info(f"Loaded {len(test_records)} test set records.")

    # Locate model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(paths["models_production"]) / "best_model.keras"

    if not model_path.exists():
        logger.warning(f"Best model not found at {model_path}. Initializing clean model for dummy evaluation.")
        model = build_retinanet_model(
            input_shape=(config["model"]["image_size"], config["model"]["image_size"], 3),
            num_classes=config["num_classes"]
        )
    else:
        logger.info(f"Loading trained model from: {model_path}")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    all_gt_boxes = []
    all_gt_classes = []

    metrics_dir = Path(paths["results_metrics"])
    preds_dir = Path(paths["results_predictions"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    image_size = config["model"].get("image_size", 512)
    conf_thresh = config["inference"].get("confidence_threshold", 0.5)

    logger.info("Running inference on test dataset samples...")
    for idx, rec in enumerate(test_records):
        img_path = rec["image_path"]
        gt_boxes = rec.get("boxes", [])
        cls_id = rec.get("class_id", 0)

        gt_cls = [cls_id] * len(gt_boxes)
        all_gt_boxes.append(gt_boxes)
        all_gt_classes.append(gt_cls)

        if Path(img_path).exists():
            try:
                img_rgb = load_image_as_rgb(img_path, target_size=(image_size, image_size))
                input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

                cls_preds, box_preds = model.predict(input_tensor, verbose=0)
                boxes, scores, classes = decode_predictions(
                    cls_preds[0], box_preds[0],
                    score_threshold=conf_thresh,
                    iou_threshold=config["inference"].get("iou_threshold", 0.5)
                )

                all_pred_boxes.append(boxes.tolist())
                all_pred_scores.append(scores.tolist())
                all_pred_classes.append(classes.tolist())

                # Save sample annotated images (first 10)
                if idx < 10:
                    annotated = draw_detections(img_rgb, boxes, scores, classes, confidence_threshold=conf_thresh)
                    out_img_path = preds_dir / f"test_pred_{idx:03d}_{rec['image_id']}.png"
                    plt.imsave(str(out_img_path), annotated)
            except Exception as e:
                logger.warning(f"Inference failed for test image {img_path}: {e}")
                all_pred_boxes.append([])
                all_pred_scores.append([])
                all_pred_classes.append([])
        else:
            all_pred_boxes.append([])
            all_pred_scores.append([])
            all_pred_classes.append([])

    # Compute evaluation metrics
    logger.info("Calculating mAP@50, mAP@50:95, IoU, and Per-Class metrics...")
    metrics_report = calculate_map_metrics(
        all_pred_boxes, all_pred_scores, all_pred_classes,
        all_gt_boxes, all_gt_classes,
        num_classes=config["num_classes"],
        class_names={int(k): v for k, v in config["classes"].items()}
    )

    # Plot Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for c_name, c_data in metrics_report["per_class"].items():
        precs = c_data.get("precisions_curve", [])
        recs = c_data.get("recalls_curve", [])
        if len(precs) > 0 and len(recs) > 0:
            ax.plot(recs, precs, label=f"{c_name} (mAP@50 = {c_data['mAP_50']:.3f})")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 1.05])
    ax.grid(True)
    ax.legend()
    pr_curve_path = metrics_dir / "precision_recall_curves.png"
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()

    # Save reports
    json_report_path = metrics_dir / "evaluation_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    txt_report_path = metrics_dir / "evaluation_report.txt"
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=== MODEL EVALUATION METRICS REPORT ===\n\n")
        f.write(f"Overall mAP@50: {metrics_report['overall_mAP_50']:.4f}\n")
        f.write(f"Overall mAP@50:95: {metrics_report['overall_mAP_50_95']:.4f}\n\n")
        for c_name, c_data in metrics_report["per_class"].items():
            f.write(f"--- Class: {c_name} ---\n")
            f.write(f"  mAP@50: {c_data['mAP_50']:.4f}\n")
            f.write(f"  mAP@50:95: {c_data['mAP_50_95']:.4f}\n")
            f.write(f"  Precision: {c_data['precision']:.4f}\n")
            f.write(f"  Recall: {c_data['recall']:.4f}\n")
            f.write(f"  True Positives (TP): {c_data['true_positives']}\n")
            f.write(f"  False Positives (FP): {c_data['false_positives']}\n")
            f.write(f"  False Negatives (FN): {c_data['false_negatives']}\n\n")

    logger.info(f"Evaluation complete. Reports generated:\n  - {json_report_path}\n  - {txt_report_path}\n  - {pr_curve_path}")
    logger.info("=== STEP 06 COMPLETE ===")

if __name__ == "__main__":
    main()

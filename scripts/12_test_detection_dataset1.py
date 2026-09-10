"""
Script 12: Test Object Detection on Dataset1 (Subset)
======================================================
Runs the trained RetinaNet detector on images picked directly from
data/raw/dataset1/val/0/  (Normal)  and  data/raw/dataset1/val/1/  (Pneumonia).
No json manifests needed — reads straight from the folder.

Usage:
    python scripts/12_test_detection_dataset1.py
    python scripts/12_test_detection_dataset1.py --max-samples 200
    python scripts/12_test_detection_dataset1.py --split test
    python scripts/12_test_detection_dataset1.py --score-threshold 0.3
"""

import sys
import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.utils import setup_logger

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Background", 1: "Pneumonia", 2: "TB"}
CLASS_COLORS = {1: "#FF4444", 2: "#44AAFF"}   # red=pneumonia, blue=tb


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_records_from_folder(dataset1_dir: Path, split: str, max_samples: int) -> list:
    """
    Read images directly from dataset1/<split>/0/  and  dataset1/<split>/1/.
    Takes the first N//2 from each class folder (sorted by filename).
    No json, no random sampling — just picks from the actual folder.
    """
    normal_dir   = dataset1_dir / split / "0"
    positive_dir = dataset1_dir / split / "1"

    if not normal_dir.exists() or not positive_dir.exists():
        raise FileNotFoundError(
            f"Expected folders:\n  {normal_dir}\n  {positive_dir}"
        )

    # Collect and sort so order is deterministic
    normal_imgs   = sorted(normal_dir.glob("*.jpg")) + sorted(normal_dir.glob("*.png"))
    positive_imgs = sorted(positive_dir.glob("*.jpg")) + sorted(positive_dir.glob("*.png"))

    half  = max_samples // 2
    n_neg = min(half, len(normal_imgs))
    n_pos = min(max_samples - n_neg, len(positive_imgs))
    n_neg = min(max_samples - n_pos, len(normal_imgs))  # re-balance if pos was short

    records = []
    for p in normal_imgs[:n_neg]:
        records.append({"image_id": p.stem, "image_path": str(p), "class_id": 0})
    for p in positive_imgs[:n_pos]:
        records.append({"image_id": p.stem, "image_path": str(p), "class_id": 1})

    return records


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the trained RetinaNet checkpoint, dynamically detecting num_classes."""
    state_dict = torch.load(model_path, map_location=device)
    num_classes = 3
    for k in state_dict.keys():
        if "head.classification_head.cls_logits.weight" in k:
            num_classes = state_dict[k].shape[0] // 9
            break
    model = retinanet_resnet50_fpn(
        weights=None,
        num_classes=num_classes,
        weights_backbone=ResNet50_Weights.DEFAULT
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path: str, image_size: int) -> torch.Tensor:
    """Load + resize + convert to [0,1] float tensor (C, H, W)."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
    except Exception:
        img = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    return torchvision.transforms.functional.to_tensor(img)


def draw_detections(img_path: str, preds: dict, score_threshold: float,
                    image_size: int, save_path: Path):
    """Save an annotated image with bounding boxes drawn on it."""
    try:
        img = Image.open(img_path).convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR
        )
    except Exception:
        img = Image.new("RGB", (image_size, image_size), (40, 40, 40))

    draw = ImageDraw.Draw(img)

    boxes  = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    n_drawn = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(int(label), "#FFFF00")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{CLASS_NAMES.get(int(label), '?')} {score:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(tag) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), tag, fill="white")
        n_drawn += 1

    img.save(save_path)
    return n_drawn


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test RetinaNet detector on images from dataset1 folder directly."
    )
    parser.add_argument("--max-samples",      type=int,   default=100,
                        help="Total images to test (50 normal + 50 positive). Default: 100.")
    parser.add_argument("--split",            type=str,   default="val",
                        choices=["val", "test", "train"],
                        help="Which dataset1 split folder to use (default: val).")
    parser.add_argument("--score-threshold",  type=float, default=0.4,
                        help="Min confidence score to keep a detection (default: 0.4).")
    parser.add_argument("--image-size",       type=int,   default=512,
                        help="Image size fed to the model (default: 512).")
    parser.add_argument("--batch-size",       type=int,   default=4,
                        help="Inference batch size (default: 4).")
    parser.add_argument("--model-path",       type=str,
                        default="models/production/best_model_pytorch.pth",
                        help="Path to the RetinaNet checkpoint.")
    parser.add_argument("--save-images",      action="store_true", default=True,
                        help="Save annotated output images (default: True).")
    parser.add_argument("--no-save-images",   dest="save_images", action="store_false",
                        help="Disable saving annotated images.")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="Random seed for sampling (default: 42).")
    parser.add_argument("--config",           type=str,   default=None)
    args = parser.parse_args()

    logger = setup_logger("test_detection_dataset1")
    logger.info("=" * 60)
    logger.info(f"  DATASET1 DETECTION TEST  (split={args.split}, n={args.max_samples})")
    logger.info("=" * 60)

    # ── Config & paths ────────────────────────────────────────
    config      = load_config(args.config)
    paths       = config["paths"]
    dataset1_dir = Path(paths["data_raw"]) / "dataset1"
    model_path  = Path(args.model_path)
    results_dir = Path(paths["results_plots"]) / "detection_test_dataset1"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset1_dir.exists():
        logger.error(f"dataset1 folder not found at {dataset1_dir}")
        sys.exit(1)
    if not model_path.exists():
        tb_model = Path("models/production/best_tbx11k_detector_pytorch.pth")
        if tb_model.exists():
            model_path = tb_model
        else:
            logger.error(f"Model not found at {model_path}")
            sys.exit(1)

    # ── Load images directly from folder ──────────────────────
    records = load_records_from_folder(dataset1_dir, args.split, args.max_samples)
    n_total = len(records)
    n_pos   = sum(1 for r in records if r["class_id"] != 0)
    n_neg   = n_total - n_pos
    logger.info(f"Loaded {n_total} images from dataset1/{args.split}/  |  Normal: {n_neg}  |  Positive: {n_pos}")
    logger.info(f"Score threshold : {args.score_threshold}")
    logger.info(f"Image size      : {args.image_size}x{args.image_size}")

    # ── Device & Model ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"Loading model from {model_path} ...")
    model = load_model(model_path, device)
    logger.info("Model loaded successfully.")

    # ── Inference ─────────────────────────────────────────────
    results = []
    batch_size = args.batch_size

    for batch_start in range(0, n_total, batch_size):
        batch_records = records[batch_start: batch_start + batch_size]
        tensors = [
            preprocess_image(r["image_path"], args.image_size)
            for r in batch_records
        ]
        batch_tensors = [t.to(device) for t in tensors]

        with torch.no_grad():
            preds = model(batch_tensors)

        for rec, pred in zip(batch_records, preds):
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            boxes  = pred["boxes"].cpu().numpy()

            # Filter by threshold
            keep = scores >= args.score_threshold
            det_scores = scores[keep]
            det_labels = labels[keep]
            det_boxes  = boxes[keep]

            n_dets = int(keep.sum())
            detected_positive = any(l in (1, 2) for l in det_labels)

            result_entry = {
                "image_id":         rec["image_id"],
                "image_path":       rec["image_path"],
                "true_class_id":    rec["class_id"],
                "n_detections":     n_dets,
                "detected_positive": detected_positive,
                "max_score":        float(det_scores.max()) if n_dets > 0 else 0.0,
                "detections": [
                    {"label": int(l), "score": float(s),
                     "box": [float(x) for x in b]}
                    for l, s, b in zip(det_labels, det_scores, det_boxes)
                ]
            }
            results.append(result_entry)

            # Save annotated image
            if args.save_images:
                # Build a filtered pred dict for drawing
                filtered_pred = {
                    "boxes":  torch.tensor(det_boxes),
                    "labels": torch.tensor(det_labels),
                    "scores": torch.tensor(det_scores),
                }
                out_name = f"{rec['image_id']}_det.jpg"
                n_drawn = draw_detections(
                    rec["image_path"], filtered_pred,
                    args.score_threshold, args.image_size,
                    results_dir / out_name
                )

        done = min(batch_start + batch_size, n_total)
        logger.info(f"  Processed {done}/{n_total} images ...")

    # ── Summary ───────────────────────────────────────────────
    tp = sum(1 for r in results if r["true_class_id"] != 0 and r["detected_positive"])
    fp = sum(1 for r in results if r["true_class_id"] == 0 and r["detected_positive"])
    fn = sum(1 for r in results if r["true_class_id"] != 0 and not r["detected_positive"])
    tn = sum(1 for r in results if r["true_class_id"] == 0 and not r["detected_positive"])

    precision  = tp / max(1, tp + fp)
    recall     = tp / max(1, tp + fn)
    f1         = 2 * precision * recall / max(1e-6, precision + recall)
    accuracy   = (tp + tn) / max(1, n_total)

    avg_dets_positive = (
        np.mean([r["n_detections"] for r in results if r["true_class_id"] != 0])
        if n_pos > 0 else 0.0
    )
    avg_dets_normal = (
        np.mean([r["n_detections"] for r in results if r["true_class_id"] == 0])
        if n_neg > 0 else 0.0
    )

    print("\n" + "=" * 60)
    print("   DETECTION TEST RESULTS  —  dataset1 val subset")
    print("=" * 60)
    print(f"  Images tested       : {n_total}  (Normal: {n_neg} | Positive: {n_pos})")
    print(f"  Split used          : dataset1/{args.split}/")
    print(f"  Score threshold     : {args.score_threshold}")
    print("-" * 60)
    print(f"  True  Positives (TP): {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True  Negatives (TN): {tn}")
    print("-" * 60)
    print(f"  Precision           : {precision * 100:.2f}%")
    print(f"  Recall (Sensitivity): {recall * 100:.2f}%")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Accuracy            : {accuracy * 100:.2f}%")
    print("-" * 60)
    print(f"  Avg detections/img (Positive cases): {avg_dets_positive:.2f}")
    print(f"  Avg detections/img (Normal cases)  : {avg_dets_normal:.2f}")
    print("=" * 60)

    if args.save_images:
        print(f"\n  Annotated images saved to: {results_dir}")

    # ── Save JSON results ─────────────────────────────────────
    out_json = results_dir / "detection_results.json"
    summary = {
        "n_tested":      n_total,
        "n_positive":    n_pos,
        "n_normal":      n_neg,
        "score_threshold": args.score_threshold,
        "seed":          args.seed,
        "metrics": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision":  round(precision,  4),
            "recall":     round(recall,     4),
            "f1":         round(f1,         4),
            "accuracy":   round(accuracy,   4),
        },
        "per_image": results
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results JSON saved to: {out_json}")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()

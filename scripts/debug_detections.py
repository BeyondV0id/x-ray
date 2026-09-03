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
from src.utils import setup_logger
from src.dataset import load_image_as_rgb
from src.model import build_retinanet_model, decode_predictions, FocalLoss, SmoothL1Loss
from src.metrics import compute_iou
from src.visualization import draw_detections, estimate_lung_region

def main():
    parser = argparse.ArgumentParser(description="Debug detection failures, low confidence, or overlapping predictions.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for debugging.")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of images.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("debug_detections")
    logger.info("=== DEBUGGING DETECTION PREDICTIONS ===")

    config = load_config(args.config)
    paths = config["paths"]
    
    debug_out = Path(paths["results_predictions"]) / "debug"
    debug_out.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(paths["models_production"]) / "best_model.keras"

    if not model_path.exists():
        logger.warning(f"Model {model_path} not found. Building fresh model for debug structure test.")
        model = build_retinanet_model(
            input_shape=(config["model"]["image_size"], config["model"]["image_size"], 3),
            num_classes=config["num_classes"]
        )
    else:
        logger.info(f"Loading model for debugging from: {model_path}")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )

    # Gather images to debug
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.folder:
        fp = Path(args.folder)
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.dcm"]:
            image_paths.extend(list(fp.rglob(ext)))
    else:
        test_manifest = Path(paths["data_processed"]) / "test.json"
        if test_manifest.exists():
            with open(test_manifest, "r", encoding="utf-8") as f:
                recs = json.load(f)
            image_paths = [Path(r["image_path"]) for r in recs[:15] if Path(r["image_path"]).exists()]

    if not image_paths:
        logger.warning("No image paths found for debugging.")
        return

    logger.info(f"Debugging detections on {len(image_paths)} images at confidence threshold {args.confidence}...")

    image_size = config["model"]["image_size"]
    class_names = {int(k): v for k, v in config["classes"].items()}

    for img_p in image_paths:
        logger.info(f"\n--- Debugging: {img_p.name} ---")
        try:
            img_rgb = load_image_as_rgb(str(img_p), target_size=(image_size, image_size))
            input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

            cls_preds, box_preds = model.predict(input_tensor, verbose=0)
            
            # Decode at low threshold to detect low-confidence proposals
            boxes, scores, classes = decode_predictions(
                cls_preds[0], box_preds[0],
                score_threshold=0.05,
                iou_threshold=0.5
            )

            high_conf_mask = scores >= args.confidence
            low_conf_mask = (scores < args.confidence) & (scores >= 0.1)

            logger.info(f"Total proposals above 5%: {len(boxes)}")
            logger.info(f"High-confidence detections (>= {args.confidence}): {np.sum(high_conf_mask)}")
            logger.info(f"Low-confidence proposals (0.10 <= conf < {args.confidence}): {np.sum(low_conf_mask)}")

            if np.sum(high_conf_mask) == 0:
                logger.warning(f"NO DETECTION PRODUCED for {img_p.name} above threshold {args.confidence}.")

            # Check for overlapping boxes
            h_boxes = boxes[high_conf_mask]
            overlaps = []
            for i in range(len(h_boxes)):
                for j in range(i+1, len(h_boxes)):
                    iou = compute_iou(h_boxes[i], h_boxes[j])
                    if iou > 0.3:
                        overlaps.append((i, j, iou))

            if overlaps:
                logger.warning(f"OVERLAPPING DETECTIONS DETECTED ({len(overlaps)} pairs > 0.30 IoU):")
                for (i, j, iou) in overlaps:
                    logger.warning(f"  Box #{i} & Box #{j} -> IoU = {iou:.3f}")

            # Draw & Save Debug annotated image
            annotated = draw_detections(
                img_rgb, boxes, scores, classes,
                confidence_threshold=args.confidence,
                class_names=class_names
            )
            out_img = debug_out / f"debug_{img_p.stem}.png"
            plt.imsave(str(out_img), annotated)
            logger.info(f"Saved debug output to: {out_img}")

        except Exception as e:
            logger.error(f"Error debugging image {img_p}: {e}")

    logger.info("\n=== DEBUGGING COMPLETE ===")

if __name__ == "__main__":
    main()

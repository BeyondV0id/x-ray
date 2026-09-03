import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, MEDICAL_DISCLAIMER
from src.dataset import load_image_as_rgb
from src.model import build_retinanet_model, decode_predictions, FocalLoss, SmoothL1Loss
from src.visualization import draw_detections, estimate_lung_region

def run_single_inference(
    image_path: str,
    model: tf.keras.Model,
    config: dict,
    output_dir: Path,
    logger
) -> None:
    img_path = Path(image_path)
    if not img_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return

    image_size = config["model"].get("image_size", 512)
    conf_thresh = config["inference"].get("confidence_threshold", 0.5)
    class_names = {int(k): v for k, v in config["classes"].items()}

    logger.info(f"Processing X-ray image: {img_path}")
    try:
        img_rgb = load_image_as_rgb(str(img_path), target_size=(image_size, image_size))
        input_tensor = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

        cls_preds, box_preds = model.predict(input_tensor, verbose=0)
        boxes, scores, classes = decode_predictions(
            cls_preds[0], box_preds[0],
            score_threshold=conf_thresh,
            iou_threshold=config["inference"].get("iou_threshold", 0.5)
        )

        logger.info(f"--- Detections for {img_path.name} ---")
        if len(boxes) == 0:
            logger.info("No disease abnormalities detected above confidence threshold.")
        else:
            for i in range(len(boxes)):
                score = float(scores[i])
                cls_id = int(classes[i])
                disease = class_names.get(cls_id, f"Class {cls_id}")
                box = boxes[i]
                lung_location = estimate_lung_region(box)

                logger.info(f"  [Detection #{i+1}]")
                logger.info(f"    - Disease: Possible {disease}")
                logger.info(f"    - Confidence: {score*100:.1f}%")
                logger.info(f"    - Bounding Box [ymin, xmin, ymax, xmax]: {np.round(box, 3).tolist()}")
                logger.info(f"    - Anatomical Location: {lung_location}")

        annotated = draw_detections(img_rgb, boxes, scores, classes, confidence_threshold=conf_thresh, class_names=class_names)
        out_file = output_dir / f"pred_{img_path.stem}.png"
        plt.imsave(str(out_file), annotated)
        logger.info(f"Annotated visualization saved to: {out_file}")

    except Exception as e:
        logger.error(f"Failed inference on {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Chest X-ray object detection inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to single input X-ray image.")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of X-ray images.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained .keras model.")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("inference")
    logger.info("=== STEP 07: INFERENCE & LOCALIZATION ===")
    logger.info(f"MEDICAL DISCLAIMER: {MEDICAL_DISCLAIMER}")

    config = load_config(args.config)
    if args.confidence is not None:
        config["inference"]["confidence_threshold"] = args.confidence

    paths = config["paths"]
    out_dir = Path(paths["results_predictions"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(paths["models_production"]) / "best_model.keras"

    if not model_path.exists():
        logger.warning(f"Production model not found at {model_path}. Initializing structure for inference check.")
        model = build_retinanet_model(
            input_shape=(config["model"]["image_size"], config["model"]["image_size"], 3),
            num_classes=config["num_classes"]
        )
    else:
        logger.info(f"Loading production model from: {model_path}")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )

    if args.image:
        run_single_inference(args.image, model, config, out_dir, logger)
    elif args.folder:
        folder_p = Path(args.folder)
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.dcm"]
        files = []
        for ext in image_extensions:
            files.extend(list(folder_p.rglob(ext)))

        logger.info(f"Found {len(files)} image files in folder: {folder_p}")
        for img_f in files:
            run_single_inference(str(img_f), model, config, out_dir, logger)
    else:
        logger.warning("No --image or --folder specified. Run with e.g. --image sample_xray.png")

    logger.info("=== STEP 07 COMPLETE ===")

if __name__ == "__main__":
    main()

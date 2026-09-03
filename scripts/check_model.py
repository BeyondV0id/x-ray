import sys
import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger
from src.model import build_retinanet_model, decode_predictions, FocalLoss, SmoothL1Loss

def main():
    parser = argparse.ArgumentParser(description="Quick sanity check for trained model loading and output signatures.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("check_model")
    logger.info("=== SANITY CHECK: MODEL SIGNATURE & INFERENCE ===")

    config = load_config(args.config)
    paths = config["paths"]
    
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(paths["models_production"]) / "best_model.keras"

    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Instantiating fresh model for signature verification.")
        model = build_retinanet_model(
            input_shape=(config["model"]["image_size"], config["model"]["image_size"], 3),
            num_classes=config["num_classes"]
        )
    else:
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )

    logger.info("\n--- Model Signature Details ---")
    logger.info(f"Model Name: {model.name}")
    logger.info(f"Input Shape: {model.input_shape}")
    logger.info(f"Output Shapes: {[out.shape for out in model.outputs]}")

    # Run dummy test inference
    img_size = config["model"]["image_size"]
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, img_size, img_size, 3)).astype(np.float32)

    logger.info("\n--- Running 1 Test Inference ---")
    cls_preds, box_preds = model.predict(dummy_input, verbose=0)
    logger.info(f"Raw Classification Output Shape: {cls_preds.shape}")
    logger.info(f"Raw Box Regression Output Shape: {box_preds.shape}")

    boxes, scores, classes = decode_predictions(
        cls_preds[0], box_preds[0],
        score_threshold=0.01,
        iou_threshold=0.5
    )

    logger.info(f"Decoded Bounding Boxes Count: {len(boxes)}")
    if len(boxes) > 0:
        logger.info(f"Sample Box [ymin, xmin, ymax, xmax]: {boxes[0].tolist()}")
        logger.info(f"Sample Confidence Score: {scores[0]:.4f}")
        logger.info(f"Sample Class Index: {classes[0]}")

    logger.info("Model verification check completed successfully!")

if __name__ == "__main__":
    main()

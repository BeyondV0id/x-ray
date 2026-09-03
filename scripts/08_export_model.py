import sys
import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger
from src.model import build_retinanet_model, FocalLoss, SmoothL1Loss

def main():
    parser = argparse.ArgumentParser(description="Export trained model to SavedModel and TFLite formats.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to input .keras model.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("export_model")
    logger.info("=== STEP 08: EXPORTING PRODUCTION MODEL ===")

    config = load_config(args.config)
    paths = config["paths"]
    prod_dir = Path(paths["models_production"])
    prod_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        input_model_path = Path(args.model_path)
    else:
        input_model_path = prod_dir / "best_model.keras"

    if not input_model_path.exists():
        logger.warning(f"Model file {input_model_path} not found. Building fresh model for export validation.")
        model = build_retinanet_model(
            input_shape=(config["model"]["image_size"], config["model"]["image_size"], 3),
            num_classes=config["num_classes"]
        )
    else:
        logger.info(f"Loading model from: {input_model_path}")
        model = tf.keras.models.load_model(
            str(input_model_path),
            custom_objects={"FocalLoss": FocalLoss, "SmoothL1Loss": SmoothL1Loss}
        )

    # 1. Export SavedModel format
    saved_model_dir = prod_dir / "saved_model"
    logger.info(f"Exporting to TensorFlow SavedModel format at: {saved_model_dir}")
    model.save(str(saved_model_dir), save_format="tf")

    # Verify SavedModel loading
    try:
        reloaded_sm = tf.keras.models.load_model(str(saved_model_dir))
        dummy_input = np.zeros((1, config["model"]["image_size"], config["model"]["image_size"], 3), dtype=np.float32)
        out_cls, out_box = reloaded_sm.predict(dummy_input, verbose=0)
        logger.info(f"SavedModel reload verification SUCCESSFUL. Output shapes: Cls={out_cls.shape}, Box={out_box.shape}")
    except Exception as e:
        logger.error(f"SavedModel reload verification failed: {e}")

    # 2. Export TFLite format
    tflite_path = prod_dir / "model.tflite"
    logger.info(f"Converting and exporting to TensorFlow Lite at: {tflite_path}")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        logger.info(f"TFLite model successfully exported ({len(tflite_model) / (1024*1024):.2f} MB).")

        # Verify TFLite loading
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        logger.info("TFLite Interpreter allocation SUCCESSFUL.")
    except Exception as e:
        logger.error(f"TFLite conversion failed or restricted for complex op graph: {e}")

    logger.info("=== STEP 08 COMPLETE ===")

if __name__ == "__main__":
    main()

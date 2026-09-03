import sys
import os
import argparse
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Download and prepare pretrained model weights.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("download_models")
    logger.info("=== STEP 00: DOWNLOAD/PREPARE PRETRAINED MODELS ===")

    config = load_config(args.config)
    pretrained_dir = Path(config["paths"]["models_pretrained"])
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = config["model"].get("backbone", "EfficientNetB0")
    weights_path = pretrained_dir / f"{backbone_name.lower()}_imagenet_weights.h5"

    logger.info(f"Target backbone: {backbone_name}")
    logger.info(f"Target weights path: {weights_path}")

    if weights_path.exists():
        logger.info(f"Pretrained weights already exist at {weights_path}. Skipping download.")
        return

    logger.info(f"Downloading ImageNet pretrained weights for {backbone_name} via TensorFlow Keras...")
    try:
        if backbone_name == "EfficientNetB0":
            model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet")
        else:
            model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")

        # Save weights to local pretrained folder
        model.save_weights(str(weights_path))
        logger.info(f"Successfully downloaded and saved pretrained weights to: {weights_path}")
    except Exception as e:
        logger.error(f"Failed to download/save pretrained model weights: {e}")
        sys.exit(1)

    logger.info("=== STEP 00 COMPLETE ===")

if __name__ == "__main__":
    main()

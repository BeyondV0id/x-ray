import sys
import os
import json
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed
from src.dataset import load_image_as_rgb
from src.visualization import draw_detections

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset annotations before training.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize per class.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("visualize_dataset")
    logger.info("=== VISUALIZING DATASET ANNOTATIONS ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    paths = config["paths"]
    samples_dir = Path(paths["results_dataset_samples"])
    samples_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path(paths["data_processed"])
    train_manifest = proc_dir / "train.json"

    if not train_manifest.exists():
        logger.error(f"Processed manifest missing at {train_manifest}. Run 03_prepare_dataset.py first.")
        sys.exit(1)

    with open(train_manifest, "r", encoding="utf-8") as f:
        records = json.load(f)

    pneu_records = [r for r in records if r["class_id"] == 0 and len(r["boxes"]) > 0]
    tb_records = [r for r in records if r["class_id"] == 1 and len(r["boxes"]) > 0]

    logger.info(f"Available annotated samples: Pneumonia={len(pneu_records)}, Tuberculosis={len(tb_records)}")

    # Sample randomly
    pneu_samples = random.sample(pneu_records, min(args.num_samples, len(pneu_records))) if pneu_records else []
    tb_samples = random.sample(tb_records, min(args.num_samples, len(tb_records))) if tb_records else []

    selected = pneu_samples + tb_samples
    class_names = {int(k): v for k, v in config["classes"].items()}
    image_size = config["model"].get("image_size", 512)

    logger.info(f"Generating visualization for {len(selected)} sample X-rays...")

    for idx, rec in enumerate(selected):
        img_p = rec["image_path"]
        boxes = np.array(rec["boxes"])
        cls_id = rec["class_id"]
        c_name = class_names.get(cls_id, f"Class {cls_id}")

        if not Path(img_p).exists():
            continue

        try:
            img_rgb = load_image_as_rgb(img_p, target_size=(image_size, image_size))
            scores = np.ones(len(boxes))  # Ground truth annotations have 100% score
            classes = np.full(len(boxes), cls_id)

            annotated = draw_detections(
                img_rgb, boxes, scores, classes,
                confidence_threshold=0.1,
                class_names=class_names,
                draw_disclaimer=True
            )

            out_file = samples_dir / f"gt_sample_{idx:02d}_{c_name}_{rec['image_id']}.png"
            plt.imsave(str(out_file), annotated)
            logger.info(f"  Saved sample annotation visualization: {out_file.name}")

        except Exception as e:
            logger.warning(f"Failed to visualize sample {img_p}: {e}")

    logger.info(f"Dataset annotation visualization completed. Output saved in: {samples_dir}")

if __name__ == "__main__":
    main()

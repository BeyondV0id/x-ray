import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed
from src.dataset import prepare_rsna_dataset, prepare_tbx11k_dataset, create_split_datasets

def main():
    parser = argparse.ArgumentParser(description="Prepare and split unified dataset for TensorFlow training.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("prepare_dataset")
    logger.info("=== STEP 03: PREPARING & SPLITTING UNIFIED DATASETS ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    paths = config["paths"]
    proc_dir = Path(paths["data_processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load RSNA (Class 0: Pneumonia)
    logger.info("Parsing RSNA Pneumonia annotations...")
    rsna_records = prepare_rsna_dataset(paths["data_pneumonia"])

    # Load TBX11K (Class 1: Tuberculosis)
    logger.info("Parsing TBX11K Tuberculosis annotations...")
    tb_records = prepare_tbx11k_dataset(paths["data_tuberculosis"])

    # Combine records
    combined_records = rsna_records + tb_records
    logger.info(f"Combined total dataset size: {len(combined_records)} images.")

    # Split dataset avoiding patient-level data leakage
    train_records, val_records, test_records = create_split_datasets(
        combined_records,
        train_ratio=config["training"].get("train_split", 0.70),
        val_ratio=config["training"].get("val_split", 0.15),
        test_ratio=config["training"].get("test_split", 0.15),
        seed=config["training"].get("seed", 42)
    )

    # Save splits to JSON manifests
    splits = {
        "train.json": train_records,
        "val.json": val_records,
        "test.json": test_records
    }

    for fname, data in splits.items():
        out_path = proc_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {fname}: {len(data)} records -> {out_path}")

    # Generate dataset metadata summary
    def count_class(records, cid):
        return sum(1 for r in records if r["class_id"] == cid)

    def count_boxes(records, cid):
        return sum(len(r["boxes"]) for r in records if r["class_id"] == cid)

    summary = {
        "total_images": len(combined_records),
        "total_annotations": sum(len(r["boxes"]) for r in combined_records),
        "class_breakdown": {
            "0_Pneumonia": {
                "images": count_class(combined_records, 0),
                "boxes": count_boxes(combined_records, 0)
            },
            "1_Tuberculosis": {
                "images": count_class(combined_records, 1),
                "boxes": count_boxes(combined_records, 1)
            }
        },
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        }
    }

    summary_file = proc_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset metadata summary saved to: {summary_file}")
    logger.info("=== STEP 03 COMPLETE ===")

if __name__ == "__main__":
    main()

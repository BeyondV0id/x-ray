import sys
import os
import json
import argparse
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, set_seed

def parse_split_dir(split_path: Path, source_name: str):
    """
    Scans a split folder (train/val/test) containing class folders '0' and '1'.
    Returns a list of image record dicts.
    """
    records = []
    if not split_path.exists():
        return records

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".dcm"}
    
    # Check subfolders 0 and 1
    for class_folder in ["0", "1"]:
        c_dir = split_path / class_folder
        if not c_dir.exists():
            continue
            
        class_id = int(class_folder)
        for img_file in c_dir.iterdir():
            if img_file.suffix.lower() in valid_exts:
                # Try getting dimensions or default to 512
                width, height = 512, 512
                records.append({
                    "image_id": img_file.stem,
                    "image_path": str(img_file.resolve()),
                    "patient_id": img_file.stem,
                    "class_id": class_id,
                    "boxes": [],
                    "width": width,
                    "height": height,
                    "source": source_name
                })
                
    return records

def main():
    parser = argparse.ArgumentParser(description="Prepare pre-split dataset in data/raw for training.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset folder.")
    parser.add_argument("--dataset-name", type=str, default="dataset1", help="Dataset folder name (e.g., dataset1, dataset2).")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("prepare_archive")
    logger.info("=== STEP 03: PREPARING PRE-SPLIT DATASET IN DATA/RAW ===")

    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    raw_path = Path(args.raw_dir).resolve()
    target_ds_dir = raw_path / args.dataset_name
    
    # Fallback to data_zips/archive if not found in data/raw
    if not target_ds_dir.exists():
        archive_path = Path("data_zips/archive").resolve() / args.dataset_name
        if archive_path.exists():
            target_ds_dir = archive_path

    if not target_ds_dir.exists():
        logger.error(f"Target dataset directory does not exist: {target_ds_dir}")
        sys.exit(1)

    logger.info(f"Processing dataset from: {target_ds_dir}")

    # Parse train, val, test
    train_records = parse_split_dir(target_ds_dir / "train", args.dataset_name)
    val_records = parse_split_dir(target_ds_dir / "val", args.dataset_name)
    test_records = parse_split_dir(target_ds_dir / "test", args.dataset_name)

    logger.info(f"Found splits -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    paths = config["paths"]
    proc_dir = Path(paths["data_processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)

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

    # Create summary metadata
    def count_class(recs, cid):
        return sum(1 for r in recs if r["class_id"] == cid)

    summary = {
        "dataset_name": args.dataset_name,
        "total_images": len(train_records) + len(val_records) + len(test_records),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "train_class_breakdown": {
            "0_Normal": count_class(train_records, 0),
            "1_Positive": count_class(train_records, 1)
        },
        "val_class_breakdown": {
            "0_Normal": count_class(val_records, 0),
            "1_Positive": count_class(val_records, 1)
        },
        "test_class_breakdown": {
            "0_Normal": count_class(test_records, 0),
            "1_Positive": count_class(test_records, 1)
        }
    }

    summary_file = proc_dir / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset metadata summary saved to: {summary_file}")
    logger.info("=== STEP 03 (ARCHIVE) COMPLETE ===")

if __name__ == "__main__":
    main()

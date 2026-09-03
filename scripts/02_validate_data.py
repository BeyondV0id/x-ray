import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger
from src.dataset import prepare_rsna_dataset, prepare_tbx11k_dataset

def main():
    parser = argparse.ArgumentParser(description="Validate raw dataset files and annotations.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("validate_data")
    logger.info("=== STEP 02: VALIDATING RAW DATASETS ===")

    config = load_config(args.config)
    paths = config["paths"]

    val_dir = Path(paths["results_validation"])
    val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate RSNA Pneumonia Dataset
    logger.info("--- Validating RSNA Pneumonia Dataset ---")
    rsna_records = prepare_rsna_dataset(paths["data_pneumonia"])
    
    rsna_positive = sum(1 for r in rsna_records if len(r["boxes"]) > 0)
    rsna_negative = len(rsna_records) - rsna_positive
    rsna_total_boxes = sum(len(r["boxes"]) for r in rsna_records)
    rsna_missing_files = sum(1 for r in rsna_records if not Path(r["image_path"]).exists())
    
    rsna_invalid_boxes = 0
    for r in rsna_records:
        for b in r["boxes"]:
            ymin, xmin, ymax, xmax = b
            if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > 1 or xmax > 1:
                rsna_invalid_boxes += 1

    logger.info(f"RSNA Total Images: {len(rsna_records)}")
    logger.info(f"RSNA Positive Cases (Pneumonia): {rsna_positive}")
    logger.info(f"RSNA Negative Cases (Normal/Other): {rsna_negative}")
    logger.info(f"RSNA Total Bounding Boxes: {rsna_total_boxes}")
    logger.info(f"RSNA Missing Image Files: {rsna_missing_files}")
    logger.info(f"RSNA Invalid Bounding Boxes: {rsna_invalid_boxes}")

    # 2. Validate TBX11K Tuberculosis Dataset
    logger.info("--- Validating TBX11K Tuberculosis Dataset ---")
    tb_records = prepare_tbx11k_dataset(paths["data_tuberculosis"])
    
    tb_positive = sum(1 for r in tb_records if len(r["boxes"]) > 0)
    tb_negative = len(tb_records) - tb_positive
    tb_total_boxes = sum(len(r["boxes"]) for r in tb_records)
    tb_missing_files = sum(1 for r in tb_records if not Path(r["image_path"]).exists())
    
    tb_invalid_boxes = 0
    for r in tb_records:
        for b in r["boxes"]:
            ymin, xmin, ymax, xmax = b
            if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > 1 or xmax > 1:
                tb_invalid_boxes += 1

    logger.info(f"TBX11K Total Images: {len(tb_records)}")
    logger.info(f"TBX11K Positive Cases (Tuberculosis): {tb_positive}")
    logger.info(f"TBX11K Negative Cases: {tb_negative}")
    logger.info(f"TBX11K Total Bounding Boxes: {tb_total_boxes}")
    logger.info(f"TBX11K Missing Image Files: {tb_missing_files}")
    logger.info(f"TBX11K Invalid Bounding Boxes: {tb_invalid_boxes}")

    report = {
        "rsna_pneumonia": {
            "total_images": len(rsna_records),
            "positive_cases": rsna_positive,
            "negative_cases": rsna_negative,
            "total_bounding_boxes": rsna_total_boxes,
            "missing_files": rsna_missing_files,
            "invalid_boxes": rsna_invalid_boxes
        },
        "tbx11k_tuberculosis": {
            "total_images": len(tb_records),
            "positive_cases": tb_positive,
            "negative_cases": tb_negative,
            "total_bounding_boxes": tb_total_boxes,
            "missing_files": tb_missing_files,
            "invalid_boxes": tb_invalid_boxes
        },
        "overall_summary": {
            "total_xray_images": len(rsna_records) + len(tb_records),
            "total_annotations": rsna_total_boxes + tb_total_boxes,
            "validation_passed": (rsna_missing_files == 0 and tb_missing_files == 0 and rsna_invalid_boxes == 0 and tb_invalid_boxes == 0)
        }
    }

    json_path = val_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    txt_path = val_dir / "validation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== DATASET VALIDATION REPORT ===\n\n")
        f.write(f"RSNA Pneumonia Images: {len(rsna_records)} (Pos: {rsna_positive}, Neg: {rsna_negative}, BBoxes: {rsna_total_boxes})\n")
        f.write(f"TBX11K Tuberculosis Images: {len(tb_records)} (Pos: {tb_positive}, Neg: {tb_negative}, BBoxes: {tb_total_boxes})\n")
        f.write(f"Total Combined Images: {len(rsna_records) + len(tb_records)}\n")
        f.write(f"Total Combined Annotations: {rsna_total_boxes + tb_total_boxes}\n")
        f.write(f"Validation Status: {'PASSED' if report['overall_summary']['validation_passed'] else 'WARNINGS DETECTED'}\n")

    logger.info(f"Validation reports generated:\n  - {json_path}\n  - {txt_path}")
    logger.info("=== STEP 02 COMPLETE ===")

if __name__ == "__main__":
    main()

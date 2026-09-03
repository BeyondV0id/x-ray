import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger
from src.visualization import draw_detections

def main():
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis (EDA) on prepared dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("analyze_dataset")
    logger.info("=== STEP 04: EXPLORATORY DATA ANALYSIS (EDA) ===")

    config = load_config(args.config)
    paths = config["paths"]

    plots_dir = Path(paths["results_plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    proc_dir = Path(paths["data_processed"])
    train_manifest = proc_dir / "train.json"

    if not train_manifest.exists():
        logger.error(f"Processed dataset manifest not found at {train_manifest}. Run 03_prepare_dataset.py first.")
        sys.exit(1)

    with open(train_manifest, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Analyzing {len(records)} records in training manifest...")

    # Statistics calculation
    pneu_imgs = [r for r in records if r["class_id"] == 0]
    tb_imgs = [r for r in records if r["class_id"] == 1]
    
    pneu_boxes = [b for r in pneu_imgs for b in r["boxes"]]
    tb_boxes = [b for r in tb_imgs for b in r["boxes"]]

    box_widths = []
    box_heights = []
    box_areas = []
    aspect_ratios = []
    x_centers = []
    y_centers = []

    all_boxes = pneu_boxes + tb_boxes
    for b in all_boxes:
        ymin, xmin, ymax, xmax = b
        w = xmax - xmin
        h = ymax - ymin
        area = w * h
        ar = w / max(h, 1e-5)
        
        box_widths.append(w)
        box_heights.append(h)
        box_areas.append(area)
        aspect_ratios.append(ar)
        x_centers.append((xmin + xmax) / 2.0)
        y_centers.append((ymin + ymax) / 2.0)

    # 1. Plot Class & Case Distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].bar(["Pneumonia", "Tuberculosis"], [len(pneu_imgs), len(tb_imgs)], color=["#FF8C00", "#00E676"])
    ax[0].set_title("Images per Class")
    ax[0].set_ylabel("Count")

    ax[1].bar(["Pneumonia BBoxes", "TB BBoxes"], [len(pneu_boxes), len(tb_boxes)], color=["#FF6B00", "#00C853"])
    ax[1].set_title("Bounding Box Annotations per Class")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plot_class_dist = plots_dir / "class_distribution.png"
    plt.savefig(plot_class_dist, dpi=300)
    plt.close()

    # 2. Plot Bounding Box Geometry Distributions
    if all_boxes:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].hist(box_areas, bins=30, color="#2196F3", edgecolor="black")
        ax[0, 0].set_title("Bounding Box Relative Area Distribution")
        ax[0, 0].set_xlabel("Normalized Area")

        ax[0, 1].hist(aspect_ratios, bins=30, color="#9C27B0", edgecolor="black")
        ax[0, 1].set_title("Aspect Ratio (Width / Height) Distribution")
        ax[0, 1].set_xlabel("Aspect Ratio")

        # Bounding box center location 2D density
        h2d, xedges, yedges, im = ax[1, 0].hist2d(x_centers, y_centers, bins=25, cmap="magma")
        ax[1, 0].set_title("Bounding Box Center Spatial Density")
        ax[1, 0].set_xlabel("X Center (Relative)")
        ax[1, 0].set_ylabel("Y Center (Relative)")
        ax[1, 0].invert_yaxis()  # Image coordinate system top to bottom

        ax[1, 1].axis("off")

        plt.tight_layout()
        plot_geometry = plots_dir / "bbox_geometry_analysis.png"
        plt.savefig(plot_geometry, dpi=300)
        plt.close()

    # Write textual EDA report
    report_file = Path(config["paths"]["results"]) / "dataset_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== CHEST X-RAY EXPLORATORY DATA ANALYSIS REPORT ===\n\n")
        f.write(f"Total Analyzed Images: {len(records)}\n")
        f.write(f"Pneumonia Images: {len(pneu_imgs)} | Annotations: {len(pneu_boxes)}\n")
        f.write(f"Tuberculosis Images: {len(tb_imgs)} | Annotations: {len(tb_boxes)}\n")
        f.write(f"Total Bounding Boxes: {len(all_boxes)}\n")
        if all_boxes:
            f.write(f"Mean BBox Area (normalized): {np.mean(box_areas):.4f} +/- {np.std(box_areas):.4f}\n")
            f.write(f"Mean Aspect Ratio (W/H): {np.mean(aspect_ratios):.4f}\n")
            f.write(f"Mean Spatial Center (X, Y): ({np.mean(x_centers):.3f}, {np.mean(y_centers):.3f})\n")

    logger.info(f"EDA Analysis plots generated under: {plots_dir}")
    logger.info(f"Dataset EDA report written to: {report_file}")
    logger.info("=== STEP 04 COMPLETE ===")

if __name__ == "__main__":
    main()

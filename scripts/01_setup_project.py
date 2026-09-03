import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger, check_gpu_status, ensure_directories

def main():
    parser = argparse.ArgumentParser(description="Create project directory structure and verify environment.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("setup_project")
    logger.info("=== STEP 01: SETTING UP PROJECT DIRECTORY STRUCTURE ===")

    config = load_config(args.config)
    paths = config["paths"]

    # Mandatory directory list
    required_dirs = [
        paths["data_raw"],
        paths["data_processed"],
        paths["data_pneumonia"],
        paths["data_tuberculosis"],
        paths["models"],
        paths["models_checkpoints"],
        paths["models_pretrained"],
        paths["models_production"],
        "notebooks",
        paths["results"],
        paths["results_validation"],
        paths["results_plots"],
        paths["results_predictions"],
        paths["results_metrics"],
        paths["results_dataset_samples"],
        paths["logs_tensorboard"],
        "docs",
        "scripts"
    ]

    created = ensure_directories(required_dirs, logger=logger)
    logger.info(f"Directory verification complete. Newly created: {len(created)} directories.")

    # Check dependencies & environment
    logger.info("Verifying Python packages and GPU availability...")
    gpu_info = check_gpu_status(logger=logger)

    env_report = {
        "python_version": sys.version,
        "platform": sys.platform,
        "gpu_status": gpu_info,
        "created_directories": created
    }

    env_file = Path(paths["results_validation"]) / "env_info.json"
    with open(env_file, "w", encoding="utf-8") as f:
        json.dump(env_report, f, indent=2)

    logger.info(f"Environment status report saved to: {env_file}")
    logger.info("=== STEP 01 COMPLETE ===")

if __name__ == "__main__":
    main()

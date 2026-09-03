import sys
import os
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for training loss & metric monitoring.")
    parser.add_argument("--port", type=int, default=6006, help="Port to run TensorBoard server.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()

    logger = setup_logger("tensorboard")
    logger.info("=== STEP 09: LAUNCHING TENSORBOARD ===")

    config = load_config(args.config)
    log_dir = Path(config["paths"]["logs_tensorboard"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"tensorboard --logdir=\"{log_dir}\" --port={args.port}"

    logger.info(f"TensorBoard log directory: {log_dir}")
    logger.info(f"To view training & validation metrics in your browser, run:\n\n  {cmd}\n")
    logger.info(f"Or open browser at: http://localhost:{args.port}/")

    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        logger.info("TensorBoard server stopped.")
    except Exception as e:
        logger.error(f"Error starting TensorBoard process: {e}")

    logger.info("=== STEP 09 COMPLETE ===")

if __name__ == "__main__":
    main()

import os
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

MEDICAL_DISCLAIMER = (
    "This system is an AI-assisted research tool and is not intended to provide a medical diagnosis. "
    "Results should not be used as a substitute for evaluation by a qualified medical professional."
)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def setup_logger(name: str = "chest_xray_ai", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def check_gpu_status(logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Check if TensorFlow can access GPU acceleration.
    
    Returns:
        Dict detailing GPU status and available physical/logical devices.
    """
    if logger is None:
        logger = setup_logger()
        
    status = {
        "gpu_available": False,
        "device_count": 0,
        "device_names": [],
        "tf_version": "Not Installed"
    }
    
    try:
        import tensorflow as tf
        status["tf_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            status["gpu_available"] = True
            status["device_count"] = len(gpus)
            status["device_names"] = [gpu.name for gpu in gpus]
            logger.info(f"GPU Acceleration ACTIVE: Detected {len(gpus)} physical GPU(s).")
            for gpu in gpus:
                logger.info(f"  - Device: {gpu.name}")
        else:
            logger.warning("GPU Acceleration UNAVAILABLE: Training will run on CPU.")
    except Exception as e:
        logger.error(f"Error checking GPU status: {e}")
        
    return status

def ensure_directories(dir_paths: List[str], logger: logging.Logger = None) -> List[str]:
    """
    Ensure that specified directory paths exist.
    
    Args:
        dir_paths (List[str]): List of directory paths to create if missing.
        
    Returns:
        List[str]: List of newly created directory paths.
    """
    if logger is None:
        logger = setup_logger()
        
    created = []
    for path_str in dir_paths:
        p = Path(path_str)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))
            logger.info(f"Created directory: {p}")
            
    return created

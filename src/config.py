import os
import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration settings from a YAML file.
    
    Args:
        config_path (str, optional): Path to YAML config file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary with resolved absolute paths.
    """
    if config_path is None:
        target_path = DEFAULT_CONFIG_PATH
    else:
        target_path = Path(config_path)
        
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")
        
    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Resolve root path (project directory)
    project_root = target_path.resolve().parent.parent
    config["project_root"] = str(project_root)
    
    # Resolve relative paths in config to absolute paths
    if "paths" in config:
        resolved_paths = {}
        for key, rel_path in config["paths"].items():
            resolved_paths[key] = str(project_root / rel_path)
        config["paths"] = resolved_paths
        
    return config

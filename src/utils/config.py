import yaml
import os
from pathlib import Path

def load_config(config_path="setup.yml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Relative path to the config file from the project root.
    
    Returns:
        dict: The configuration dictionary.
    """
    # Find project root
    # Assuming this file is in src/utils/config.py, so root is two levels up
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    
    abs_config_path = project_root / config_path
    
    if not abs_config_path.exists():
        # Try finding it in the current working directory
        abs_config_path = Path(os.getcwd()) / config_path
        
    if not abs_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {abs_config_path}")
        
    with open(abs_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

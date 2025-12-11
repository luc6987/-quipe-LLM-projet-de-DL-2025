"""
Path Configuration Module
路径配置模块

Centralized path management for cache, models, and output directories.
集中管理缓存、模型和输出目录的路径。

All paths are now controlled by setup.yml.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from src.utils.config import load_config

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the project root directory.
    获取项目根目录。
    """
    # Start from the directory containing this file (src/utils/paths.py)
    current_dir = Path(__file__).parent.absolute()
    
    # Traverse up until we find project root markers
    for parent in [current_dir] + list(current_dir.parents):
        # Check for project root markers
        if (parent / '.git').exists() or (parent / 'README.md').exists():
            return parent
    
    # Fallback: go up 3 levels from src/utils/ to reach project root
    return current_dir.parent.parent

PROJECT_ROOT = get_project_root()

# Load paths from config or use defaults
try:
    config = load_config()
    paths_config = config.get('paths', {})
    
    # Define relative paths from config
    output_rel = paths_config.get('output_dir', 'output')
    cache_rel = paths_config.get('cache_dir', 'output/cache')
    hf_rel = paths_config.get('huggingface_dir', 'output/huggingface')
    
    logger.info(f"Loaded paths from config: OUTPUT={output_rel}, CACHE={cache_rel}, HF={hf_rel}")
    
except Exception as e:
    logger.warning(f"Could not load paths from setup.yml, using defaults: {e}")
    output_rel = 'output'
    cache_rel = 'output/cache'
    hf_rel = 'output/huggingface'

# Resolve absolute paths
OUTPUT_DIR = PROJECT_ROOT / output_rel
CACHE_DIR = PROJECT_ROOT / cache_rel
HF_CACHE_DIR = PROJECT_ROOT / hf_rel

# Subdirectories
MODELS_CACHE_DIR = CACHE_DIR / 'models'
DATA_CACHE_DIR = CACHE_DIR / 'data'
OPTUNA_CACHE_DIR = CACHE_DIR / 'optuna'

MODELS_OUTPUT_DIR = OUTPUT_DIR / 'models'
SUBMISSIONS_DIR = OUTPUT_DIR / 'submissions'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'
SHAP_DIR = OUTPUT_DIR / 'shap'
LOGS_DIR = OUTPUT_DIR / 'logs'
# Unified preprocessing output directory
PREPROCESSING_DIR = OUTPUT_DIR / 'preprocessing'

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'


def ensure_cache_dir() -> Path:
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def ensure_output_dir() -> Path:
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_cache_dir(subdir: Optional[str] = None) -> Path:
    """Get cache directory path."""
    ensure_cache_dir()
    if subdir:
        cache_path = CACHE_DIR / subdir
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    return CACHE_DIR


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """Get output directory path."""
    ensure_output_dir()
    if subdir:
        output_path = OUTPUT_DIR / subdir
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    return OUTPUT_DIR


def get_models_cache_dir() -> Path:
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_CACHE_DIR


def get_data_cache_dir() -> Path:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE_DIR


def get_optuna_cache_dir() -> Path:
    OPTUNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return OPTUNA_CACHE_DIR


def get_models_output_dir() -> Path:
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_OUTPUT_DIR


def get_submissions_dir() -> Path:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SUBMISSIONS_DIR


def get_visualizations_dir() -> Path:
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return VISUALIZATIONS_DIR


def get_preprocessing_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the unified preprocessing output directory.
    """
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    if subdir:
        target = PREPROCESSING_DIR / subdir
        target.mkdir(parents=True, exist_ok=True)
        return target
    return PREPROCESSING_DIR


def get_preprocessing_visualizations_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the preprocessing visualizations directory (e.g., stats under preprocessing).
    """
    base = get_preprocessing_dir('visualizations')
    if subdir:
        target = base / subdir
        target.mkdir(parents=True, exist_ok=True)
        return target
    return base


def get_embeddings_dir() -> Path:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDINGS_DIR


def get_shap_dir() -> Path:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    return SHAP_DIR


def get_logs_dir() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def setup_huggingface_cache():
    """Setup HuggingFace cache environment variables."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(HF_CACHE_DIR)
    os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR / 'transformers')
    os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / 'datasets')
    os.environ['HF_HUB_CACHE'] = str(HF_CACHE_DIR / 'hub')
    
    # Also set XDG_CACHE_HOME to ensure everything goes there
    # Note: XDG_CACHE_HOME usually affects more than just HF, but we want to isolate
    # this project's cache if possible.
    os.environ['XDG_CACHE_HOME'] = str(CACHE_DIR)


# Initialize directories on import
ensure_cache_dir()
ensure_output_dir()



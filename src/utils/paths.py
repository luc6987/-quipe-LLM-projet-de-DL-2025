"""
Path Configuration Module
路径配置模块

Centralized path management for cache, models, and output directories.
集中管理缓存、模型和输出目录的路径。

All cache files (intermediate processing results, Optuna studies, etc.) should be placed in cache/.
All output files (submission files, model checkpoints, visualizations, etc.) should be placed in output/.
所有缓存文件（中间处理结果、Optuna studies 等）应放在 cache/ 目录。
所有输出文件（提交文件、模型检查点、可视化等）应放在 output/ 目录。
"""

import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    获取项目根目录。
    
    This function traverses up from the current file location to find the project root.
    The project root is identified as the directory containing .git or README.md.
    此函数从当前文件位置向上遍历以找到项目根目录。
    项目根目录通过包含 .git 或 README.md 的目录来识别。
    
    Returns:
        Path: Project root directory path
    """
    # Start from the directory containing this file (src/utils/paths.py)
    current_dir = Path(__file__).parent.absolute()
    
    # Traverse up until we find project root markers
    for parent in [current_dir] + list(current_dir.parents):
        # Check for project root markers
        if (parent / '.git').exists() or (parent / 'README.md').exists():
            return parent
    
    # Fallback: go up 3 levels from src/utils/ to reach project root
    # Default structure: project_root/src/utils/paths.py
    return current_dir.parent.parent


# Cache directories - for intermediate processing results, Optuna studies, etc.
# 缓存目录 - 用于中间处理结果、Optuna studies 等
PROJECT_ROOT = get_project_root()
CACHE_DIR = PROJECT_ROOT / 'cache'
MODELS_CACHE_DIR = CACHE_DIR / 'models'  # For cached model files
DATA_CACHE_DIR = CACHE_DIR / 'data'      # For cached data processing results
OPTUNA_CACHE_DIR = CACHE_DIR / 'optuna'  # For Optuna study databases and configs
HF_CACHE_DIR = PROJECT_ROOT / '.cache' / 'huggingface'  # HuggingFace model cache

# Output directories - for final results, submissions, checkpoints, visualizations, etc.
# 输出目录 - 用于最终结果、提交文件、检查点、可视化等
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_OUTPUT_DIR = OUTPUT_DIR / 'models'         # For trained model checkpoints
SUBMISSIONS_DIR = OUTPUT_DIR / 'submissions'      # For Kaggle submission files
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'  # For plots and visualizations
EMBEDDINGS_DIR = OUTPUT_DIR / 'embeddings'        # For embedding vectors
SHAP_DIR = OUTPUT_DIR / 'shap'                    # For SHAP analysis results
LOGS_DIR = OUTPUT_DIR / 'logs'                    # For training logs

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'


def ensure_cache_dir() -> Path:
    """
    Ensure cache directory exists and return its path.
    确保缓存目录存在并返回其路径。
    
    Returns:
        Path: Cache directory path
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def ensure_output_dir() -> Path:
    """
    Ensure output directory exists and return its path.
    确保输出目录存在并返回其路径。
    
    Returns:
        Path: Output directory path
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_cache_dir(subdir: Optional[str] = None) -> Path:
    """
    Get cache directory path, optionally with a subdirectory.
    获取缓存目录路径，可选择包含子目录。
    
    Args:
        subdir: Optional subdirectory name within cache/
        
    Returns:
        Path: Cache directory path (with subdirectory if specified)
    """
    ensure_cache_dir()
    if subdir:
        cache_path = CACHE_DIR / subdir
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    return CACHE_DIR


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """
    Get output directory path, optionally with a subdirectory.
    获取输出目录路径，可选择包含子目录。
    
    Args:
        subdir: Optional subdirectory name within output/
        
    Returns:
        Path: Output directory path (with subdirectory if specified)
    """
    ensure_output_dir()
    if subdir:
        output_path = OUTPUT_DIR / subdir
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    return OUTPUT_DIR


def get_models_cache_dir() -> Path:
    """Get models cache directory. 获取模型缓存目录。"""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_CACHE_DIR


def get_data_cache_dir() -> Path:
    """Get data cache directory. 获取数据缓存目录。"""
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE_DIR


def get_optuna_cache_dir() -> Path:
    """Get Optuna cache directory. 获取 Optuna 缓存目录。"""
    OPTUNA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return OPTUNA_CACHE_DIR


def get_models_output_dir() -> Path:
    """Get models output directory. 获取模型输出目录。"""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_OUTPUT_DIR


def get_submissions_dir() -> Path:
    """Get submissions directory. 获取提交文件目录。"""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SUBMISSIONS_DIR


def get_visualizations_dir() -> Path:
    """Get visualizations directory. 获取可视化目录。"""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return VISUALIZATIONS_DIR


def get_embeddings_dir() -> Path:
    """Get embeddings directory. 获取嵌入向量目录。"""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return EMBEDDINGS_DIR


def get_shap_dir() -> Path:
    """Get SHAP analysis directory. 获取 SHAP 分析目录。"""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    return SHAP_DIR


def get_logs_dir() -> Path:
    """Get logs directory. 获取日志目录。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def setup_huggingface_cache():
    """
    Setup HuggingFace cache environment variables.
    设置 HuggingFace 缓存环境变量。
    
    This should be called before importing any HuggingFace libraries.
    此函数应在导入任何 HuggingFace 库之前调用。
    """
    # Ensure cache directory exists
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set all HuggingFace related environment variables
    os.environ['HF_HOME'] = str(HF_CACHE_DIR)
    os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR / 'transformers')
    os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / 'datasets')
    os.environ['HF_HUB_CACHE'] = str(HF_CACHE_DIR / 'hub')
    
    # Also set XDG_CACHE_HOME to override default cache location
    xdg_cache = PROJECT_ROOT / '.cache'
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(xdg_cache)


# Initialize directories on import
# 导入时初始化目录
ensure_cache_dir()
ensure_output_dir()
get_models_cache_dir()
get_data_cache_dir()
get_optuna_cache_dir()
get_models_output_dir()
get_submissions_dir()
get_visualizations_dir()
get_embeddings_dir()
get_shap_dir()
get_logs_dir()


"""
Utility Functions Module
工具函数模块 / Module de fonctions utilitaires / Utility Functions Module / Hilfsfunktionen-Modul

Provides essential utility functions for data loading, GPU detection, and file operations.
提供数据加载、GPU检测和文件操作的基础工具函数。
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# 添加 /Data 目录中的包路径（如果存在）
# Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

# GPU detection imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_gpu_for_xgboost() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Detect GPU availability for XGBoost and return appropriate parameters.
    检测XGBoost的GPU可用性并返回相应参数。
    Détecte la disponibilité GPU pour XGBoost et retourne les paramètres appropriés.
    Erkennt GPU-Verfügbarkeit für XGBoost und gibt entsprechende Parameter zurück.
    
    This function checks CUDA availability and tests XGBoost GPU support.
    For XGBoost 2.0+, uses tree_method='hist' with device='cuda'.
    
    Returns:
        Tuple containing:
        - gpu_available (bool): Whether GPU is available and working
        - tree_method (str): Tree method to use ('hist' for both CPU/GPU in 2.0+)
        - device_params (dict): Dictionary with device parameters for XGBoost
            - If GPU available: {'device': 'cuda', 'tree_method': 'hist'}
            - If CPU only: {'tree_method': 'hist'} (no device key)
    
    Raises:
        ImportError: If xgboost is not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
    
    device_params = {}
    gpu_available = False
    tree_method = 'hist'
    
    # Check CUDA availability via multiple methods
    cuda_available = False
    
    # Method 1: Check via PyTorch (if available)
    if TORCH_AVAILABLE:
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                logger.info(f"✓ CUDA detected via PyTorch:")
                logger.info(f"    Device: {device_name}")
                logger.info(f"    Device count: {device_count}")
        except Exception as e:
            logger.debug(f"Could not check CUDA via PyTorch: {e}")
    
    # Method 2: Check CUDA_VISIBLE_DEVICES environment variable
    if not cuda_available:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            logger.info(f"  CUDA_VISIBLE_DEVICES={cuda_visible} detected")
            # Still need to test if XGBoost can actually use it
    
    # Method 3: Try to check nvidia-smi (optional, for diagnostics)
    if not cuda_available:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                logger.info("  nvidia-smi detected, but PyTorch CUDA check failed")
                logger.info("  Attempting XGBoost GPU test anyway...")
                # Set cuda_available to True to attempt XGBoost GPU test
                cuda_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass  # nvidia-smi not available or timed out
    
    # Test XGBoost GPU support with a small test model
    if cuda_available:
        try:
            # Create a small test dataset
            X_test = np.random.rand(100, 10).astype(np.float32)
            y_test = np.random.randint(0, 2, 100)
            
            # Try XGBoost 2.0+ API: tree_method='hist' with device='cuda'
            try:
                test_model = xgb.XGBClassifier(
                    device='cuda',
                    tree_method='hist',
                    n_estimators=1,
                    random_state=42
                )
                test_model.fit(X_test, y_test)
                gpu_available = True
                device_params = {'device': 'cuda', 'tree_method': 'hist'}
                logger.info("✓ XGBoost GPU support confirmed (XGBoost 2.0+ API)")
            except Exception as e1:
                # Try older API: tree_method='gpu_hist'
                try:
                    test_model = xgb.XGBClassifier(
                        tree_method='gpu_hist',
                        n_estimators=1,
                        random_state=42
                    )
                    test_model.fit(X_test, y_test)
                    gpu_available = True
                    tree_method = 'gpu_hist'
                    device_params = {'tree_method': 'gpu_hist'}
                    logger.info("✓ XGBoost GPU support confirmed (legacy API: gpu_hist)")
                except Exception as e2:
                    logger.warning(f"GPU detected but XGBoost GPU support failed: {e1}")
                    logger.warning("Falling back to CPU mode")
                    gpu_available = False
                    device_params = {'tree_method': 'hist'}
        except Exception as e:
            logger.warning(f"Error testing XGBoost GPU: {e}")
            logger.warning("Falling back to CPU mode")
            gpu_available = False
            device_params = {'tree_method': 'hist'}
    else:
        logger.info("⚠ No CUDA device detected, using CPU mode")
        device_params = {'tree_method': 'hist'}
        logger.info("  Tip: Ensure CUDA is properly installed and XGBoost is built with GPU support")
        logger.info("  For XGBoost GPU support, install: pip install xgboost[gpu]")
    
    if gpu_available:
        logger.info(f"✓ GPU acceleration enabled: {device_params}")
    else:
        logger.info(f"⚠ Using CPU mode: {device_params}")
        if cuda_available and not gpu_available:
            logger.info("  Note: CUDA detected but XGBoost GPU support failed")
            logger.info("  This may indicate XGBoost was not built with GPU support")
    
    return gpu_available, tree_method, device_params


def load_jsonl(file_path: str) -> pd.DataFrame:
    """
    Load and parse JSONL (JSON Lines) file into a pandas DataFrame.
    加载并解析JSONL文件为pandas DataFrame。
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame: Parsed data with nested JSON flattened
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading JSONL file: {file_path}")
    
    records = []
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    records.append(data)
                    line_count += 1
                    
                    # Progress logging for large files
                    if line_count % 10000 == 0:
                        logger.info(f"  Processed {line_count} lines...")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"  Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        if not records:
            raise ValueError(f"No valid records found in {file_path}")
        
        logger.info(f"  Successfully loaded {len(records)} records")
        
        # Flatten nested JSON structure
        from pandas import json_normalize
        df = json_normalize(records)
        
        logger.info(f"  Flattened to DataFrame shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading JSONL file: {e}")
        raise


def extract_user_id(df: pd.DataFrame) -> pd.Series:
    """
    Extract user ID from DataFrame for GroupKFold grouping.
    从DataFrame中提取用户ID用于GroupKFold分组。
    
    Strategy: Try multiple methods to extract unique user identifier.
    策略：尝试多种方法提取唯一用户标识符。
    
    Args:
        df: DataFrame with user information
        
    Returns:
        Series: User ID series for grouping
        
    Raises:
        ValueError: If no user identifier can be found
    """
    user_id = None
    
    # Method 1: Direct user.id (preferred)
    if 'user.id' in df.columns:
        user_id = df['user.id'].copy()
        logger.info("  Using 'user.id' as user identifier")
    elif 'user.id_str' in df.columns:
        user_id = df['user.id_str'].copy()
        logger.info("  Using 'user.id_str' as user identifier")
    # Method 2: Combine user.screen_name + user.created_at
    elif 'user.screen_name' in df.columns and 'user.created_at' in df.columns:
        user_id = (df['user.screen_name'].astype(str) + '_' + 
                  df['user.created_at'].astype(str))
        logger.info("  Using 'user.screen_name + user.created_at' as user identifier")
    # Method 3: Last resort - use index (not recommended, may cause data leakage)
    else:
        logger.warning("  ⚠ No user identifier found, using index (may cause data leakage!)")
        user_id = pd.Series(range(len(df)), index=df.index)
    
    if user_id is None:
        raise ValueError("Could not extract user ID from DataFrame")
    
    unique_users = user_id.nunique()
    logger.info(f"  Found {unique_users} unique users")
    
    return user_id


def save_submission(predictions: np.ndarray, ids: pd.Series, output_path: str) -> None:
    """
    Save predictions to CSV file in submission format.
    将预测结果保存为CSV提交文件。
    
    Args:
        predictions: Array of predictions (0 or 1)
        ids: Series of challenge IDs
        output_path: Path to save the submission file (will be placed in output/submissions/ if relative)
        
    Raises:
        ValueError: If predictions and ids lengths don't match
    """
    if len(predictions) != len(ids):
        raise ValueError(f"Predictions length ({len(predictions)}) doesn't match IDs length ({len(ids)})")
    
    # If output_path is relative or just a filename, place it in submissions directory
    output_path_obj = Path(output_path)
    if not output_path_obj.is_absolute():
        from src.utils.paths import get_submissions_dir
        output_path = str(get_submissions_dir() / output_path_obj.name)
    
    ensure_dir(output_path)
    
    submission_df = pd.DataFrame({
        'ID': ids,
        'Prediction': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    logger.info(f"✓ Submission saved to: {output_path}")
    logger.info(f"  Predictions distribution:")
    logger.info(f"    {submission_df['Prediction'].value_counts().sort_index().to_dict()}")


def ensure_dir(file_path: str) -> None:
    """
    Ensure the directory for a file path exists, create if necessary.
    确保文件路径的目录存在，如不存在则创建。
    
    Args:
        file_path: File path (directory will be extracted)
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


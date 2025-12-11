"""
Unified Library Imports Module
统一库导入模块

Centralizes all library imports across the project to avoid conflicts and ensure consistent initialization.
集中管理项目中所有库的导入，避免冲突并确保一致的初始化。
"""

import os
import sys
import logging

_logger = logging.getLogger(__name__)

# ============================================================================
# Path Configuration
# ============================================================================

# Remove pytorch_packages from sys.path to avoid conflicts (we'll add it back only for specific imports)
# This prevents issues with broken numpy/pandas installations in pytorch_packages
_original_path = sys.path.copy()
sys.path = [p for p in sys.path if 'pytorch_packages' not in p]

# Store pytorch_packages path for later use (if needed)
_pytorch_packages_path = '/Data/yuguang.yao/pytorch_packages' if os.path.exists('/Data/yuguang.yao/pytorch_packages') else None

# ============================================================================
# PyTorch and Transformers Imports (with directory switching fix)
# ============================================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
torch = None
CamembertTokenizer = None
CamembertForSequenceClassification = None
CamembertModel = None
AutoModel = None
AutoTokenizer = None
DebertaV2Tokenizer = None
AutoConfig = None
Trainer = None
TrainingArguments = None
EarlyStoppingCallback = None
DataCollatorWithPadding = None
AdamW = None
get_linear_schedule_with_warmup = None
SequenceClassifierOutput = None

try:
    original_cwd = os.getcwd()
    
    # Clear any existing torch modules (avoid using corrupted imports)
    torch_modules = [k for k in list(sys.modules.keys()) if k.startswith('torch')]
    for mod in torch_modules:
        del sys.modules[mod]
    
    # Switch to /tmp directory before importing PyTorch (fixes C extension loading issue)
    os.chdir('/tmp')
    
    # Import PyTorch
    import torch
    TORCH_AVAILABLE = True
    
    # Import transformers components
    from transformers import (
        CamembertTokenizer,
        CamembertModel,
        CamembertForSequenceClassification,
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup
    )
    from transformers.modeling_outputs import SequenceClassifierOutput
    
    # Try to import AdamW (may be in different location in different versions)
    try:
        from transformers import AdamW
    except ImportError:
        try:
            from transformers.optimization import AdamW
        except ImportError:
            try:
                from torch.optim import AdamW
            except ImportError:
                AdamW = None
    
    # Try to import DeBERTa tokenizer (may not be available)
    try:
        from transformers import DebertaV2Tokenizer
    except ImportError:
        DebertaV2Tokenizer = None
    
    TRANSFORMERS_AVAILABLE = True
    
except Exception as e:
    TORCH_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False
    torch = None
    # Only log warning if it's not the expected C extension error
    if 'Failed to load PyTorch C extensions' not in str(e) and 'cudaGetDriverEntryPointByVersion' not in str(e):
        _logger.warning(f"PyTorch/transformers not available: {e}. Deep learning features will be skipped.")
finally:
    # Ensure we restore the original working directory
    try:
        os.chdir(original_cwd)
    except:
        pass

# ============================================================================
# PyTorch Utilities
# ============================================================================

if TORCH_AVAILABLE and torch is not None:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    torch_nn = nn
    torch_F = F
    torch_Dataset = Dataset
    torch_DataLoader = DataLoader
else:
    torch_nn = None
    torch_F = None
    torch_Dataset = None
    torch_DataLoader = None

# ============================================================================
# XGBoost Import
# ============================================================================

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    _logger.warning("XGBoost not available. XGBoost features will be skipped.")

# ============================================================================
# Scikit-learn Imports
# ============================================================================

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    sklearn_PCA = PCA
    sklearn_accuracy_score = accuracy_score
    sklearn_GroupKFold = GroupKFold
    sklearn_StratifiedKFold = StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn_PCA = None
    sklearn_accuracy_score = None
    sklearn_GroupKFold = None
    sklearn_StratifiedKFold = None
    _logger.warning("scikit-learn not available. Some features will be skipped.")

# ============================================================================
# Optuna Import
# ============================================================================

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# ============================================================================
# Evaluate Library Import
# ============================================================================

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    evaluate = None

# ============================================================================
# SHAP Import (with special handling)
# ============================================================================

SHAP_AVAILABLE = False
shap = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    # Try importing from pytorch_packages if standard import failed
    if _pytorch_packages_path:
        try:
            import importlib.util
            shap_path = os.path.join(_pytorch_packages_path, 'shap')
            if os.path.exists(shap_path):
                spec = importlib.util.spec_from_file_location("shap", os.path.join(shap_path, "__init__.py"))
                if spec and spec.loader:
                    shap = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(shap)
                    SHAP_AVAILABLE = True
        except Exception:
            pass

# ============================================================================
# Matplotlib and Seaborn Imports
# ============================================================================

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

# ============================================================================
# Standard Library Imports (always available)
# ============================================================================

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# ============================================================================
# Export Summary
# ============================================================================

__all__ = [
    # PyTorch
    'TORCH_AVAILABLE', 'torch', 'torch_nn', 'torch_F', 'torch_Dataset', 'torch_DataLoader',
    # Transformers
    'TRANSFORMERS_AVAILABLE', 'CamembertTokenizer', 'CamembertModel', 
    'CamembertForSequenceClassification', 'AutoModel', 'AutoTokenizer', 
    'AutoConfig', 'DebertaV2Tokenizer', 'Trainer', 'TrainingArguments',
    'EarlyStoppingCallback', 'DataCollatorWithPadding', 'AdamW',
    'get_linear_schedule_with_warmup', 'SequenceClassifierOutput',
    # XGBoost
    'XGBOOST_AVAILABLE', 'xgb',
    # Scikit-learn
    'SKLEARN_AVAILABLE', 'sklearn_PCA', 'sklearn_accuracy_score',
    'sklearn_GroupKFold', 'sklearn_StratifiedKFold',
    # Other libraries
    'OPTUNA_AVAILABLE', 'optuna',
    'EVALUATE_AVAILABLE', 'evaluate',
    'SHAP_AVAILABLE', 'shap',
    'MATPLOTLIB_AVAILABLE', 'plt',
    'SEABORN_AVAILABLE', 'sns',
    # Standard libraries
    'np', 'pd', 'json', 'pickle', 'Path', 'Dict', 'Any', 'Optional', 
    'List', 'Tuple', 'datetime', 'logging'
]


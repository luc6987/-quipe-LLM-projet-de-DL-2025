"""
SHAP Analysis Module
SHAP分析模块 / Module d'analyse SHAP / SHAP Analysis Module / SHAP-Analysemodul

Implements Phase 4: Feature Interpretability using SHAP (SHapley Additive exPlanations).
实现第四阶段：使用SHAP进行特征可解释性分析 / Implements Phase 4: Feature Interpretability using SHAP (SHapley Additive exPlanations)

SHAP Values:
- Based on Shapley values from cooperative game theory
- Quantifies each feature's contribution to prediction
- Provides local (per-sample) and global (aggregate) interpretability

Etymology: "Shapley" from Lloyd Shapley (1923-2016), Nobel Prize winner in Economics
for work on cooperative game theory and fair value distribution.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging

# 添加 /Data 目录中的包路径（如果存在）
# Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

# Try importing SHAP from multiple locations
SHAP_AVAILABLE = False
shap = None

# Method 1: Try standard import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Method 2: Try importing from pytorch_packages if standard import failed
if not SHAP_AVAILABLE and os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    try:
        import importlib.util
        shap_path = '/Data/yuguang.yao/pytorch_packages/shap'
        if os.path.exists(shap_path):
            spec = importlib.util.spec_from_file_location("shap", os.path.join(shap_path, "__init__.py"))
            if spec and spec.loader:
                shap = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(shap)
                SHAP_AVAILABLE = True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not import SHAP from pytorch_packages: {e}")

# Method 3: Try importing as a package in pytorch_packages
if not SHAP_AVAILABLE:
    try:
        import importlib
        shap = importlib.import_module('shap')
        SHAP_AVAILABLE = True
    except ImportError:
        pass

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Initialize logger after imports
logger = logging.getLogger(__name__)

# Log SHAP availability status
if SHAP_AVAILABLE:
    try:
        logger.debug(f"SHAP imported successfully, version: {getattr(shap, '__version__', 'unknown')}")
    except:
        pass
else:
    logger.debug("SHAP not available - will skip SHAP analysis")


def analyze_with_shap(model: Any, X_val: pd.DataFrame, feature_names: Optional[List[str]] = None,
                      max_samples: int = 1000, max_display: int = 20,
                      output_dir: str = 'output/shap', plot: bool = True) -> Dict[str, Any]:
    """
    Perform SHAP analysis for feature interpretability.
    执行SHAP分析以进行特征可解释性分析。
    
    Generates:
    1. SHAP summary plot (beeswarm plot) - shows feature importance and impact
    2. Feature importance ranking based on mean absolute SHAP values
    
    Args:
        model: Trained XGBoost model
        X_val: Validation set features DataFrame
        feature_names: List of feature names (if None, uses DataFrame columns)
        max_samples: Maximum number of samples for SHAP calculation (for speed)
        max_display: Maximum number of features to display in plots
        output_dir: Directory to save SHAP plots and CSV files
        plot: Whether to generate visualization plots
        
    Returns:
        Dictionary containing:
        - 'shap_values': SHAP values array
        - 'feature_importance': DataFrame with feature importance rankings
        - 'explainer': SHAP TreeExplainer object
    """
    if not SHAP_AVAILABLE or shap is None:
        error_msg = "SHAP is not available. "
        if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
            error_msg += "SHAP may be installed in /Data/yuguang.yao/pytorch_packages but could not be imported. "
        error_msg += "Please ensure SHAP is properly installed: pip install shap"
        raise ImportError(error_msg)
    
    logger.info("="*60)
    logger.info("SHAP Analysis: Feature Interpretability")
    logger.info("="*60)
    
    # Convert to numpy array
    if isinstance(X_val, pd.DataFrame):
        X_val_array = X_val.values
        if feature_names is None:
            feature_names = X_val.columns.tolist()
    else:
        X_val_array = X_val
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_val_array.shape[1])]
    
    # Limit samples for faster computation
    if len(X_val_array) > max_samples:
        logger.info(f"  Limiting to {max_samples} samples for faster SHAP computation")
        sample_indices = np.random.choice(len(X_val_array), max_samples, replace=False)
        X_val_array = X_val_array[sample_indices]
    else:
        sample_indices = None
    
    # Create SHAP TreeExplainer for XGBoost
    # TreeExplainer is optimized for tree-based models (XGBoost, LightGBM, etc.)
    logger.info("  Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    logger.info("  Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_val_array)
    
    # Handle binary classification: SHAP returns list for binary, single array for regression
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    logger.info(f"  ✓ SHAP values computed: shape {shap_values.shape}")
    
    # Calculate feature importance (mean absolute SHAP value)
    # This is more reliable than XGBoost's built-in importance for interpretability
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    logger.info(f"\n  Top {min(max_display, len(feature_importance))} Important Features:")
    for idx, row in feature_importance.head(max_display).iterrows():
        logger.info(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Generate visualizations
    if plot and MATPLOTLIB_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Summary Plot (Beeswarm Plot)
            # Shows feature importance and impact direction
            logger.info("\n  Generating SHAP summary plot (beeswarm)...")
            plt.figure(figsize=(10, max(6, max_display * 0.4)))
            shap.summary_plot(
                shap_values,
                X_val_array,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            summary_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
            plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"  ✓ Summary plot saved: {summary_plot_path}")
            
            # 2. Bar Plot (Feature Importance)
            logger.info("  Generating SHAP bar plot...")
            plt.figure(figsize=(10, max(6, max_display * 0.4)))
            shap.plots.bar(
                shap_values,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            bar_plot_path = os.path.join(output_dir, 'shap_bar_plot.png')
            plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"  ✓ Bar plot saved: {bar_plot_path}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Error generating plots: {e}")
    
    # Save feature importance to CSV
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_csv_path, index=False)
    logger.info(f"  ✓ Feature importance saved: {importance_csv_path}")
    
    logger.info(f"{'='*60}")
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'explainer': explainer,
        'sample_indices': sample_indices
    }


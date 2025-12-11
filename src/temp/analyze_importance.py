"""
Feature Importance Analysis Module
特征重要性分析模块

Extracts and analyzes feature importance from the trained classifier.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    model: Any,  # xgb.XGBClassifier
    feature_names: list,
    top_n: int = 50,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Extract and analyze feature importance from trained XGBoost model.
    
    Args:
        model: Trained XGBoost classifier
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save importance CSV. If None, uses default.
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available.")
    
    logger.info("="*60)
    logger.info("Feature Importance Analysis")
    logger.info("="*60)
    
    # Get feature importance (gain)
    importance_gain = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_gain
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Display top N features
    logger.info(f"\nTop {top_n} Features for Distinguishing Train vs Test:")
    logger.info("-" * 60)
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']:50s}  {row['importance']:.6f}")
    
    # Statistics
    logger.info(f"\nImportance Statistics:")
    logger.info(f"  Total features: {len(importance_df)}")
    logger.info(f"  Mean importance: {importance_df['importance'].mean():.6f}")
    logger.info(f"  Std importance: {importance_df['importance'].std():.6f}")
    logger.info(f"  Max importance: {importance_df['importance'].max():.6f}")
    logger.info(f"  Min importance: {importance_df['importance'].min():.6f}")
    
    # Features with zero importance
    zero_importance = (importance_df['importance'] == 0).sum()
    if zero_importance > 0:
        logger.info(f"  Features with zero importance: {zero_importance}")
    
    # Cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    importance_df['cumulative_pct'] = importance_df['cumulative_importance'] / importance_df['importance'].sum() * 100
    
    # Find how many features account for 80% of importance
    top_80_pct = (importance_df['cumulative_pct'] <= 80).sum()
    logger.info(f"  Top {top_80_pct} features account for 80% of total importance")
    
    # Save to CSV
    if save_path is None:
        from src.utils.paths import get_output_dir
        output_dir = get_output_dir('temp')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'feature_importance.csv'
    
    logger.info(f"\nSaving feature importance to {save_path}...")
    importance_df.to_csv(save_path, index=False)
    logger.info("Feature importance saved.")
    
    return importance_df


def get_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    min_importance: Optional[float] = None
) -> list:
    """
    Get list of top N features or features above minimum importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to return
        min_importance: Minimum importance threshold (optional)
        
    Returns:
        List of feature names
    """
    if min_importance is not None:
        top_features = importance_df[importance_df['importance'] >= min_importance]['feature'].tolist()
    else:
        top_features = importance_df.head(top_n)['feature'].tolist()
    
    return top_features


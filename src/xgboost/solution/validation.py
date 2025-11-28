"""
Validation Module
验证模块 / Module de validation / Validation Module / Validierungsmodul

Implements Phase 2: Robust Validation Strategy using GroupKFold.
实现第二阶段：使用GroupKFold的稳健验证策略。

Critical: MUST use GroupKFold to prevent data leakage.
关键：必须使用GroupKFold防止数据泄露。
Data Leakage / 数据泄露 / Fuite de données / Datenleck occurs when tweets from
the same user are split between train and validation sets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


def group_kfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   model_params: Dict[str, Any], n_splits: int = 5,
                   use_gpu: bool = True, gpu_params: Dict[str, Any] = None) -> Tuple[List[float], List[int]]:
    """
    Perform GroupKFold cross-validation to prevent data leakage.
    执行GroupKFold交叉验证以防止数据泄露。
    
    Why GroupKFold?
    - Labels are user-level, but data is tweet-level
    - Same user's tweets MUST be in same fold (train or validation)
    - Prevents leakage: model can't see same user's tweets in both sets
    
    GroupKFold原理：
    - 标签是用户级别的，但数据是推文级别的
    - 同一用户的所有推文必须在同一折中（训练集或验证集）
    - 防止泄露：模型不能在训练集和验证集中看到同一用户的推文
    
    Args:
        X: Feature DataFrame
        y: Target labels Series
        groups: User ID Series for grouping (prevents data leakage)
        model_params: XGBoost model parameters dictionary
        n_splits: Number of CV folds (default: 5)
        use_gpu: Whether to use GPU acceleration
        gpu_params: GPU parameters dict from utils.detect_gpu_for_xgboost()
        
    Returns:
        Tuple of:
        - cv_scores: List of accuracy scores for each fold
        - best_iterations: List of best iteration numbers for each fold
          (used to determine final n_estimators)
    """
    logger.info("="*60)
    logger.info(f"GroupKFold Cross-Validation (n_splits={n_splits})")
    logger.info("="*60)
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    if isinstance(groups, pd.Series):
        groups_array = groups.values
    else:
        groups_array = groups
    
    # Validate groups
    unique_users = len(np.unique(groups_array))
    logger.info(f"  Unique users: {unique_users}")
    logger.info(f"  Total samples: {len(X_array)}")
    logger.info(f"  Average tweets per user: {len(X_array) / unique_users:.2f}")
    
    # Initialize GroupKFold
    # GroupKFold ensures all samples from same group are in same fold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Prepare model parameters with GPU settings
    final_params = model_params.copy()
    if use_gpu and gpu_params:
        final_params.update(gpu_params)
        logger.info(f"  Using GPU acceleration: {gpu_params}")
    else:
        final_params['tree_method'] = 'hist'
        logger.info("  Using CPU mode")
    
    cv_scores = []
    best_iterations = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_array, y_array, groups=groups_array), 1):
        logger.info(f"\nFold {fold}/{n_splits}:")
        
        X_train_fold = X_array[train_idx]
        X_val_fold = X_array[val_idx]
        y_train_fold = y_array[train_idx]
        y_val_fold = y_array[val_idx]
        
        # Verify no user overlap between train and validation
        train_users = set(groups_array[train_idx])
        val_users = set(groups_array[val_idx])
        overlap = train_users & val_users
        if overlap:
            logger.error(f"  ⚠ ERROR: User overlap detected! {len(overlap)} users in both sets")
            raise ValueError("Data leakage detected: users appear in both train and validation sets")
        else:
            logger.debug(f"  ✓ No user overlap (train: {len(train_users)}, val: {len(val_users)})")
        
        # Create model with parameters
        # Note: early_stopping_rounds must be set in model initialization for XGBoost 2.1.4+
        model_params_fold = final_params.copy()
        model_params_fold['early_stopping_rounds'] = 50
        
        model = xgb.XGBClassifier(**model_params_fold)
        
        # Train with early stopping
        logger.info(f"  Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(acc)
        
        # Get best iteration (for determining final n_estimators)
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            best_iter = model.best_iteration
        else:
            # Fallback: use n_estimators if best_iteration not available
            best_iter = model_params_fold.get('n_estimators', 100)
        
        best_iterations.append(best_iter)
        
        logger.info(f"  ✓ Fold {fold} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        logger.info(f"  ✓ Best iteration: {best_iter}")
    
    # Summary statistics
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    mean_iterations = np.mean(best_iterations)
    
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Results:")
    logger.info(f"  Mean Accuracy: {mean_score:.4f} ({mean_score*100:.2f}%)")
    logger.info(f"  Std Deviation: {std_score:.4f} ({std_score*100:.2f}%)")
    logger.info(f"  Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"  Mean Best Iterations: {mean_iterations:.1f}")
    logger.info(f"{'='*60}")
    
    return cv_scores, best_iterations







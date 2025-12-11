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

# 从统一导入模块导入所有库
from typing import Callable, Optional

from src._imports import (
    np, pd, Tuple, List, Dict, Any, logging,
    sklearn_GroupKFold as GroupKFold,
    sklearn_accuracy_score as accuracy_score,
    xgb
)

logger = logging.getLogger(__name__)


def group_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    use_gpu: bool = True,
    gpu_params: Dict[str, Any] = None,
    model_type: str = "xgboost",
    model_builder: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Tuple[List[float], List[int]]:
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
    elif model_type == "xgboost":
        final_params.setdefault('tree_method', 'hist')
        logger.info("  Using CPU mode (XGBoost hist)")
    elif model_type == "catboost":
        final_params.setdefault('task_type', 'CPU')
        logger.info("  Using CPU mode (CatBoost)")
    else:
        final_params.setdefault('device', 'cpu')
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
        
        model_params_fold = final_params.copy()
        logger.info(f"  Training on {len(train_idx)} samples, validating on {len(val_idx)} samples...")

        if model_type == "xgboost":
            model_params_fold.setdefault('early_stopping_rounds', 50)
            model = model_builder(model_params_fold) if model_builder else xgb.XGBClassifier(**model_params_fold)
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
        elif model_type == "catboost":
            if model_builder:
                model = model_builder(model_params_fold)
            else:
                try:
                    from catboost import CatBoostClassifier
                except ImportError as exc:
                    raise ImportError("CatBoost is not installed but model_type='catboost' was requested.") from exc
                model = CatBoostClassifier(**model_params_fold)
            
            # CatBoost uses eval_set and early_stopping_rounds in fit()
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
                early_stopping_rounds=50
            )
        else:  # model_type == "lightgbm"
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError("LightGBM is not installed but model_type='lightgbm' was requested.") from exc

            if model_builder:
                model = model_builder(model_params_fold)
            else:
                model = lgb.LGBMClassifier(**model_params_fold)

            fit_kwargs = {
                "eval_set": [(X_val_fold, y_val_fold)],
            }
            if "eval_metric" not in model_params_fold:
                fit_kwargs["eval_metric"] = "binary_logloss"
            try:
                callbacks = []
                if hasattr(lgb, "early_stopping"):
                    callbacks.append(lgb.early_stopping(50, verbose=False))
                fit_kwargs["callbacks"] = callbacks
            except Exception:
                pass

            model.fit(X_train_fold, y_train_fold, **fit_kwargs)
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(acc)
        
        # Get best iteration (for determining final n_estimators/iterations)
        if model_type == "xgboost":
            best_iter = getattr(model, 'best_iteration', None)
        elif model_type == "catboost":
            # CatBoost stores best iteration in best_iteration_ attribute
            best_iter = getattr(model, 'best_iteration_', None) or getattr(model, 'best_iteration', None)
        else:  # model_type == "lightgbm"
            best_iter = getattr(model, 'best_iteration_', None) or getattr(model, 'best_iteration', None)

        if best_iter is None:
            # Fallback to configured iterations/n_estimators
            if model_type == "catboost":
                best_iter = model_params_fold.get('iterations', 100)
            else:
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







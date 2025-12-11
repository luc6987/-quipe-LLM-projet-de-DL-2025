"""
CatBoost helpers to mirror the LightGBM/XGBoost utilities.
"""

from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_gpu_for_catboost() -> Tuple[bool, Dict[str, Any]]:
    """
    Lightweight GPU detection for CatBoost.
    Returns availability flag and gpu-related params suitable for CatBoostClassifier.
    """
    try:
        import catboost  # noqa: F401
    except ImportError:
        logger.warning("CatBoost not installed; GPU check skipped.")
        return False, {}

    # CatBoost automatically detects GPU if available
    # We can check via environment variable or try a simple test
    import os

    use_gpu_env = os.environ.get("CATBOOST_FORCE_GPU", "0")
    if use_gpu_env == "1":
        logger.info("CATBOOST_FORCE_GPU=1 detected, forcing GPU mode.")
        return True, {"task_type": "GPU", "devices": "0"}

    # Check if CUDA is available (similar to LightGBM check)
    try:
        from src._imports import TORCH_AVAILABLE, torch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("CUDA detected, CatBoost will use GPU if available.")
            return True, {"task_type": "GPU", "devices": "0"}
    except Exception:
        pass

    # Default: assume GPU available only when explicitly requested by caller.
    return False, {}


def build_catboost_params(params: Dict[str, Any], use_gpu: bool, gpu_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge user params with GPU hints."""
    final_params = params.copy()
    if use_gpu and gpu_params:
        final_params.update(gpu_params)
    elif use_gpu:
        # Default to GPU if caller asked for it but we have no hints.
        final_params.setdefault("task_type", "GPU")
        final_params.setdefault("devices", "0")
    else:
        final_params.setdefault("task_type", "CPU")
    
    # GPU mode doesn't support rsm (feature sampling) for non-pairwise modes
    # Remove rsm if using GPU
    if final_params.get("task_type") == "GPU" and "rsm" in final_params:
        logger.warning("rsm parameter is not supported on GPU for non-pairwise modes. Removing rsm parameter.")
        final_params.pop("rsm", None)
    
    return final_params


def create_catboost_classifier(params: Dict[str, Any]):
    """Factory to create a CatBoost classifier with provided params."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Please install catboost to use the CatBoost branch."
        ) from exc

    # Filter out XGBoost/LightGBM-specific parameters that CatBoost doesn't support
    catboost_params = params.copy()
    # Remove XGBoost/LightGBM-specific params if present
    unsupported_params = [
        'scale_pos_weight_ratio', 'tree_method', 'gpu_id', 'early_stopping_rounds',
        'device', 'gpu_platform_id', 'gpu_device_id',  # LightGBM GPU params
        'boosting_type',  # LightGBM specific
        'num_leaves',  # LightGBM specific
        'min_child_samples',  # LightGBM specific
        'colsample_bytree',  # XGBoost/LightGBM - CatBoost uses 'rsm' instead
    ]
    for param in unsupported_params:
        catboost_params.pop(param, None)
    
    # Map parameter names to CatBoost equivalents
    # XGBoost/LightGBM uses 'eval_metric', CatBoost uses 'eval_metric' in fit() but 'loss_function' in constructor
    if 'eval_metric' in catboost_params and 'loss_function' not in catboost_params:
        # CatBoost uses 'loss_function' for the main objective
        # 'eval_metric' can be passed to fit() but we'll keep it for now
        pass
    
    # Map 'colsample_bytree' to 'rsm' (random subspace method)
    if 'colsample_bytree' in params and 'rsm' not in catboost_params:
        catboost_params['rsm'] = params['colsample_bytree']
        catboost_params.pop('colsample_bytree', None)
    
    # Map 'min_child_weight' to 'min_data_in_leaf' (CatBoost equivalent)
    if 'min_child_weight' in params and 'min_data_in_leaf' not in catboost_params:
        catboost_params['min_data_in_leaf'] = params['min_child_weight']
        catboost_params.pop('min_child_weight', None)
    
    # Map 'reg_alpha' to 'l2_leaf_reg' (CatBoost uses different naming)
    if 'reg_alpha' in params and 'l2_leaf_reg' not in catboost_params:
        # Note: CatBoost's l2_leaf_reg is similar but not exactly the same
        # We'll use reg_lambda if available, otherwise reg_alpha
        if 'reg_lambda' in params:
            catboost_params['l2_leaf_reg'] = params['reg_lambda']
        else:
            catboost_params['l2_leaf_reg'] = params['reg_alpha']
        catboost_params.pop('reg_alpha', None)
        catboost_params.pop('reg_lambda', None)
    
    # Ensure iterations is set (CatBoost uses 'iterations' instead of 'n_estimators')
    if 'n_estimators' in catboost_params and 'iterations' not in catboost_params:
        catboost_params['iterations'] = catboost_params.pop('n_estimators')
    elif 'iterations' not in catboost_params:
        catboost_params.setdefault('iterations', 100)
    
    # Set default loss_function for binary classification if not specified
    if 'loss_function' not in catboost_params and 'objective' not in catboost_params:
        catboost_params.setdefault('loss_function', 'Logloss')
    
    # Map 'objective' to 'loss_function' if needed
    if 'objective' in catboost_params and 'loss_function' not in catboost_params:
        obj = catboost_params.pop('objective')
        if obj == 'binary' or obj == 'binary:logistic':
            catboost_params['loss_function'] = 'Logloss'
        else:
            catboost_params['loss_function'] = obj
    
    # CatBoost uses 'learning_rate' (same as others)
    # CatBoost uses 'depth' instead of 'max_depth'
    if 'max_depth' in catboost_params and 'depth' not in catboost_params:
        catboost_params['depth'] = catboost_params.pop('max_depth')
    
    # CatBoost uses 'subsample' but requires 'bootstrap_type' to be 'Bernoulli'
    # Default bootstrap type 'Bayesian' doesn't support subsample
    if 'subsample' in catboost_params and 'bootstrap_type' not in catboost_params:
        catboost_params['bootstrap_type'] = 'Bernoulli'
    
    # CatBoost uses 'random_seed' instead of 'random_state'
    if 'random_state' in catboost_params and 'random_seed' not in catboost_params:
        catboost_params['random_seed'] = catboost_params.pop('random_state')
    
    # CatBoost uses 'thread_count' instead of 'n_jobs'
    if 'n_jobs' in catboost_params and 'thread_count' not in catboost_params:
        n_jobs = catboost_params.pop('n_jobs')
        if n_jobs == -1:
            import os
            catboost_params['thread_count'] = os.cpu_count() or 1
        else:
            catboost_params['thread_count'] = n_jobs
    
    # Set verbose=False by default to reduce output
    if 'verbose' not in catboost_params:
        catboost_params['verbose'] = False
    
    return CatBoostClassifier(**catboost_params)


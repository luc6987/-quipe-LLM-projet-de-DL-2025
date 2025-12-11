"""
LightGBM helpers to mirror the XGBoost utilities.
"""

from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_gpu_for_lightgbm() -> Tuple[bool, Dict[str, Any]]:
    """
    GPU detection for LightGBM.
    Returns availability flag and gpu-related params suitable for LGBMClassifier.
    
    Similar to XGBoost/CatBoost, this function actively detects GPU availability.
    """
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        logger.warning("LightGBM not installed; GPU check skipped.")
        return False, {}

    import os

    # Check environment variable override first
    use_gpu_env = os.environ.get("LIGHTGBM_FORCE_GPU", "")
    if use_gpu_env == "1":
        logger.info("LIGHTGBM_FORCE_GPU=1 detected, forcing GPU mode.")
        return True, {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    elif use_gpu_env == "0":
        logger.info("LIGHTGBM_FORCE_GPU=0 detected, forcing CPU mode.")
        return False, {}

    # Active GPU detection (similar to XGBoost/CatBoost)
    cuda_available = False
    
    # Method 1: Check via PyTorch (if available)
    try:
        from src._imports import TORCH_AVAILABLE, torch
        if TORCH_AVAILABLE:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                logger.info(f"✓ CUDA detected via PyTorch for LightGBM:")
                logger.info(f"    Device: {device_name}")
                logger.info(f"    Device count: {device_count}")
    except Exception as e:
        logger.debug(f"Could not check CUDA via PyTorch: {e}")

    # Method 2: Check CUDA_VISIBLE_DEVICES environment variable
    if not cuda_available:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and cuda_visible != '-1':
            logger.info(f"  CUDA_VISIBLE_DEVICES={cuda_visible} detected")
            # Set cuda_available to True to attempt LightGBM GPU test
            cuda_available = True

    # Method 3: Try to check nvidia-smi (optional, for diagnostics)
    if not cuda_available:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                logger.info("  nvidia-smi detected, but PyTorch CUDA check failed")
                logger.info("  Attempting LightGBM GPU test anyway...")
                cuda_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass  # nvidia-smi not available or timed out

    # Test LightGBM GPU support with a small test model
    gpu_available = False
    if cuda_available:
        try:
            from lightgbm import LGBMClassifier
            # Create a small test dataset
            import numpy as np
            X_test = np.random.rand(100, 10).astype(np.float32)
            y_test = np.random.randint(0, 2, 100)
            
            # Try LightGBM GPU
            try:
                test_model = LGBMClassifier(
                    device='gpu',
                    n_estimators=1,
                    random_state=42,
                    verbose=-1
                )
                test_model.fit(X_test, y_test)
                gpu_available = True
                logger.info("✓ LightGBM GPU support confirmed")
            except Exception as e:
                logger.warning(f"LightGBM GPU test failed: {e}")
                logger.warning("Falling back to CPU mode for LightGBM")
                gpu_available = False
        except Exception as e:
            logger.warning(f"Error testing LightGBM GPU: {e}")
            logger.warning("Falling back to CPU mode for LightGBM")
            gpu_available = False
    else:
        logger.info("⚠ No CUDA device detected for LightGBM, using CPU mode")

    if gpu_available:
        gpu_params = {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0
        }
        logger.info(f"✓ LightGBM GPU acceleration enabled: {gpu_params}")
        return True, gpu_params
    else:
        logger.info(f"⚠ LightGBM using CPU mode")
        return False, {}


def build_lgbm_params(params: Dict[str, Any], use_gpu: bool, gpu_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge user params with GPU hints."""
    final_params = params.copy()
    if use_gpu and gpu_params:
        final_params.update(gpu_params)
    elif use_gpu:
        # Default to GPU if caller asked for it but we have no hints.
        final_params.setdefault("device", "gpu")
        final_params.setdefault("gpu_platform_id", 0)
        final_params.setdefault("gpu_device_id", 0)
    else:
        final_params.setdefault("device", "cpu")
    return final_params


def create_lgbm_classifier(params: Dict[str, Any]):
    """Factory to create a LightGBM classifier with provided params."""
    try:
        # Direct import from system lightgbm package (no conflict since we renamed local package)
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Please install lightgbm to use the LightGBM branch."
        ) from exc

    # Filter out XGBoost-specific parameters that LightGBM doesn't support
    lgbm_params = params.copy()
    # Remove XGBoost-specific params if present
    xgb_only_params = ['scale_pos_weight_ratio', 'tree_method', 'gpu_id', 'early_stopping_rounds']
    for param in xgb_only_params:
        lgbm_params.pop(param, None)
    
    # Map XGBoost param names to LightGBM equivalents if needed
    # XGBoost uses 'eval_metric' in fit(), LightGBM uses 'metric' in constructor
    if 'eval_metric' in lgbm_params and 'metric' not in lgbm_params:
        eval_metric = lgbm_params.pop('eval_metric')
        if isinstance(eval_metric, list):
            lgbm_params['metric'] = eval_metric
        else:
            lgbm_params['metric'] = [eval_metric] if isinstance(eval_metric, str) else eval_metric
    
    # Ensure n_estimators is set (LightGBM uses n_estimators, same as XGBoost)
    if 'n_estimators' not in lgbm_params:
        lgbm_params.setdefault('n_estimators', 100)
    
    return LGBMClassifier(**lgbm_params)


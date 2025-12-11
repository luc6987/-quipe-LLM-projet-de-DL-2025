#!/usr/bin/env python3
"""
Main Pipeline Script
 / Script de pipeline principal / Main Pipeline Script / Haupt-Pipeline-Skript

Complete end-to-end pipeline for Kaggle "Influencer or Observer" classification.
Kaggle "Influencer or Observer"  / Complete end-to-end pipeline for Kaggle "Influencer or Observer" classification challenge

Workflow:
1. GPU Detection
2. Data Loading (JSONL parsing)
3. Feature Engineering (metadata only, NO text content)
4. User ID Extraction (for GroupKFold)
5. Hyperparameter Optimization (Optuna with GPU)
6. Final Training (on full training set with GPU)
7. SHAP Analysis (feature interpretability)
8. Test Set Processing (using training transformers)
9. Submission Generation (CSV format)

Model switch:
- Configure `model.active` in `setup.yml` to choose `xgboost`, `lightgbm`, or `catboost`.
- Each model has its own parameter block (`xgboost`, `lightgbm`, `catboost`) and uses the same preprocessing/CV/shap steps.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all libraries from unified imports module
from src._imports import (
    np, pd, json, pickle, datetime, logging,
    Dict, Any, Optional,
    xgb, XGBOOST_AVAILABLE
)
from src.lgbm_integration.utils import (
    build_lgbm_params,
    create_lgbm_classifier,
    detect_gpu_for_lightgbm,
)
from src.catboost_integration.utils import (
    build_catboost_params,
    create_catboost_classifier,
    detect_gpu_for_catboost,
)

# Import solution modules
from src.xgboost.utils import (
    detect_gpu_for_xgboost,
    load_jsonl,
    extract_user_id,
    save_submission,
    ensure_dir
)
from src.preprocessing.xgboost_feature_engineering import engineer_features
from src.xgboost.validation import group_kfold_cv
from src.xgboost.hyperparameter_optimization import optimize_hyperparameters
from src.vis_stat.training_monitor import generate_training_report
from src.vis_stat.error_analysis import log_error_statistics

# Import shared baseline-style data processing utilities
from src.preprocessing import data_processing as dp
from src.preprocessing.camembert_preprocessing import extract_and_fuse_text_fields
from src.utils.config import load_config
from src.utils.paths import get_preprocessing_dir
# SHAP analysis imported lazily (only when needed) to handle import errors gracefully

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M%S'
)
logger = logging.getLogger(__name__)

PREPROCESSING_FLAG = "preprocessing_complete.json"


def _pick_metric_key(metric_dict: Dict[str, Any]) -> Optional[str]:
    """Pick a logloss-like metric key if available, otherwise the first key."""
    if not metric_dict:
        return None
    keys = list(metric_dict.keys())
    for k in keys:
        if "logloss" in k.lower():
            return k
    return keys[0]


def _normalize_train_val_curves(evals_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract train/validation logloss curves from various evals_result formats.
    Returns dict with train, val, metric_name if found.
    """
    if not evals_result:
        return None

    datasets = list(evals_result.keys())
    if not datasets:
        return None

    # Heuristics to find train/val keys across xgboost/lightgbm/catboost
    preferred_train = ["train", "learn", "training", "validation_0", "valid_0", "validation0"]
    preferred_val = ["validation_1", "valid_1", "validation", "val", "validation1", "test"]

    train_key = next((k for k in preferred_train if k in evals_result), None)
    val_key = next((k for k in preferred_val if k in evals_result), None)

    if train_key is None:
        train_key = datasets[0]
    if val_key is None and len(datasets) > 1:
        val_key = datasets[1]

    metric_name = None
    if train_key in evals_result:
        metric_name = _pick_metric_key(evals_result[train_key])
    if metric_name is None and val_key and val_key in evals_result:
        metric_name = _pick_metric_key(evals_result[val_key])
    if metric_name is None:
        return None

    train_vals = evals_result.get(train_key, {}).get(metric_name, [])
    val_vals = evals_result.get(val_key, {}).get(metric_name, []) if val_key else []

    # Ensure list of floats
    train_vals = [float(v) for v in train_vals]
    val_vals = [float(v) for v in val_vals]

    if not train_vals and not val_vals:
        return None

    return {
        "train": train_vals,
        "val": val_vals,
        "metric": metric_name,
        "train_key": train_key,
        "val_key": val_key,
    }


def _save_and_plot_boosting_summary(
    evals_result: Dict[str, Any],
    monitoring_base: Path,
    report_dir: Path,
    model_name: str,
) -> None:
    """
    Persist current model's eval curves and generate a combined boosting summary plot.
    One color per model; solid = train, dashed = val.
    """
    summary_dir = monitoring_base
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_train_val_curves(evals_result)
    if normalized is None:
        logger.warning("No train/val curves found to aggregate for %s", model_name)
        return

    # Save current model curves for later aggregation
    model_json_path = summary_dir / f"{model_name}_evals_result.json"
    with open(model_json_path, "w", encoding="utf-8") as f_out:
        json.dump(normalized, f_out, indent=2)

    # Load all available model curves
    model_colors = {
        "xgboost": "#1f77b4",
        "lightgbm": "#ff7f0e",
        "catboost": "#2ca02c",
    }
    available_models = []
    for mdl in ("xgboost", "lightgbm", "catboost"):
        path = summary_dir / f"{mdl}_evals_result.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    available_models.append((mdl, data))
            except Exception as exc:
                logger.warning("Failed to load curves for %s: %s", mdl, exc)

    if not available_models:
        logger.warning("No model curves available to plot summary.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Matplotlib unavailable, skip summary plot: %s", exc)
        return

    plt.figure(figsize=(10, 6))
    for mdl, data in available_models:
        color = model_colors.get(mdl, None)
        train = data.get("train", [])
        val = data.get("val", [])
        metric_name = data.get("metric", "loss")
        epochs_train = range(1, len(train) + 1)
        epochs_val = range(1, len(val) + 1)
        if train:
            plt.plot(epochs_train, train, label=f"{mdl} train", color=color, linestyle="-")
        if val:
            plt.plot(epochs_val, val, label=f"{mdl} val", color=color, linestyle="--")
        plt.ylabel(metric_name)

    plt.title("Boosting Models - Train/Validation Logloss")
    plt.xlabel("Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    summary_png = summary_dir / "training_curve_logloss_overview.png"
    report_png = report_dir / "boosting_training_loss_overview.png"
    plt.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.savefig(report_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved boosting summary plot to %s and %s", summary_png, report_png)


def _get_preprocessing_paths(base_dir: Path) -> Dict[str, Path]:
    """Define all preprocessing artifact paths under a single directory."""
    return {
        "base": base_dir,
        "train_text_raw": base_dir / "train_text_raw.pkl",
        "train_nontext_raw": base_dir / "train_nontext_raw.pkl",
        "test_text_raw": base_dir / "test_text_raw.pkl",
        "test_nontext_raw": base_dir / "test_nontext_raw.pkl",
        "camembert_train_texts": base_dir / "camembert_train_texts.pkl",
        "camembert_test_texts": base_dir / "camembert_test_texts.pkl",
        "train_features": base_dir / "train_features.pkl",
        "test_features": base_dir / "test_features.pkl",
        "train_labels": base_dir / "train_labels.pkl",
        "test_ids": base_dir / "test_ids.pkl",
        "train_groups": base_dir / "train_groups.pkl",
        "metadata": base_dir / "metadata.json",
        "flag": base_dir / PREPROCESSING_FLAG,
    }


def _load_preprocessed_artifacts(paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    """Load cached preprocessing artifacts if they all exist."""
    required_keys = ["train_features", "test_features", "train_labels", "test_ids", "train_groups", "flag"]
    if not all(paths[k].exists() for k in required_keys):
        return None
    try:
        X_train = pd.read_pickle(paths["train_features"])
        X_test = pd.read_pickle(paths["test_features"])
        y_train = pd.read_pickle(paths["train_labels"])
        test_ids = pd.read_pickle(paths["test_ids"])
        train_groups = pd.read_pickle(paths["train_groups"])
        logger.info("Loaded preprocessed artifacts from %s", paths["base"])
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "test_ids": test_ids,
            "train_groups": train_groups,
        }
    except Exception as exc:
        logger.warning("Failed to load cached preprocessing artifacts: %s", exc)
        return None


def _save_preprocessing_artifacts(
    paths: Dict[str, Path],
    train_text_df: Optional[pd.DataFrame],
    train_nontext_df: Optional[pd.DataFrame],
    test_text_df: Optional[pd.DataFrame],
    test_nontext_df: Optional[pd.DataFrame],
    camembert_train_texts: Optional[pd.Series],
    camembert_test_texts: Optional[pd.Series],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    train_groups: pd.Series,
    metadata: Dict[str, Any],
) -> None:
    """Persist preprocessing outputs for reuse."""
    paths["base"].mkdir(parents=True, exist_ok=True)

    if train_text_df is not None:
        train_text_df.to_pickle(paths["train_text_raw"])
    if train_nontext_df is not None:
        train_nontext_df.to_pickle(paths["train_nontext_raw"])
    if test_text_df is not None:
        test_text_df.to_pickle(paths["test_text_raw"])
    if test_nontext_df is not None:
        test_nontext_df.to_pickle(paths["test_nontext_raw"])

    if camembert_train_texts is not None:
        camembert_train_texts.to_pickle(paths["camembert_train_texts"])
    if camembert_test_texts is not None:
        camembert_test_texts.to_pickle(paths["camembert_test_texts"])

    X_train.to_pickle(paths["train_features"])
    X_test.to_pickle(paths["test_features"])
    y_train.to_pickle(paths["train_labels"])
    test_ids.to_pickle(paths["test_ids"])
    train_groups.to_pickle(paths["train_groups"])

    metadata_to_save = {
        **metadata,
        "created_at": datetime.now().isoformat(),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f_meta:
        json.dump(metadata_to_save, f_meta, indent=2, ensure_ascii=False)
    with open(paths["flag"], "w", encoding="utf-8") as f_flag:
        json.dump({"status": "complete", **metadata_to_save}, f_flag, indent=2, ensure_ascii=False)
    logger.info("Saved preprocessing artifacts to %s", paths["base"])


def main():
    """Main execution function."""
    # Use unified path configuration
    from src.utils.paths import PROJECT_ROOT, DATA_DIR, get_submissions_dir, get_shap_dir
    
    default_train = str(DATA_DIR / 'train.jsonl')
    default_test = str(DATA_DIR / 'kaggle_test.jsonl')
    default_output = str(get_submissions_dir() / 'submission.csv')
    default_shap_dir = str(get_shap_dir())
    
    parser = argparse.ArgumentParser(
        description='Complete XGBoost Pipeline for Influencer/Observer Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--train-file', type=str, default=default_train,
                       help='Path to training JSONL file')
    parser.add_argument('--test-file', type=str, default=default_test,
                       help='Path to test JSONL file')
    parser.add_argument('--output', type=str, default=default_output,
                       help='Path to output submission CSV file')
    
    # Optimization options
    parser.add_argument('--optimize', action='store_true',
                       help='Run Optuna hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of Optuna optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Optimization timeout in seconds (None = no limit)')
    
    # GPU options
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                       help='Disable GPU acceleration')
    
    # CV options
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--cv-strategy', type=str, default='stratified', choices=['stratified', 'group'],
                       help='CV strategy: stratified (allows user overlap) or group (prevents leakage)')
    
    # Leaderboard-oriented mode
    parser.add_argument(
        '--kaggle-max',
        action='store_true',
        help='Enable leaderboard-oriented mode: keep user-level IDs/names as categorical '
             'features to maximize Kaggle LB (introduces user-level leakage).',
    )
    
    # SHAP options
    parser.add_argument('--shap', action='store_true', default=True,
                       help='Run SHAP analysis (default: True)')
    parser.add_argument('--no-shap', dest='shap', action='store_false',
                       help='Skip SHAP analysis')
    parser.add_argument('--shap-samples', type=int, default=1000,
                       help='Maximum samples for SHAP analysis')
    parser.add_argument('--shap-output-dir', type=str, default=default_shap_dir,
                       help='Directory for SHAP output files')
    
    # Kaggle submission options
    parser.add_argument('--submit', action='store_true',
                       help='Submit to Kaggle and print score')
    parser.add_argument('--competition', type=str, 
                       default='influencers-or-observers-predicting-social-roles',
                       help='Kaggle competition name')
    parser.add_argument('--submission-message', type=str, default='XGBoost submission',
                       help='Message for Kaggle submission')
    
    # Optimization target options
    parser.add_argument('--use-kaggle-score', action='store_true',
                       help='Use Kaggle submission score for optimization instead of CV score')
    parser.add_argument('--kaggle-wait-time', type=int, default=300,
                       help='Max wait time (seconds) for Kaggle score after submission (default: 300)')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config()
        semantic_config = config.get('semantic', {})
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
        config = {}
        semantic_config = {}
    prep_conf = config.get('preprocessing', {})
    text_columns = prep_conf.get('text_columns', ['text', 'extended_tweet.full_text'])
    
    # Check if stacking mode is enabled
    stacking_config = config.get('stacking', {})
    if stacking_config.get('enabled', False):
        logger.info("="*80)
        logger.info("Stacking Mode Enabled")
        logger.info("="*80)
        logger.info("Running stacking ensemble instead of single model training...")
        try:
            from src.stacking import run_stacking
            run_stacking()
            logger.info("Stacking completed successfully!")
            return
        except Exception as e:
            logger.error(f"Stacking failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    model_config = config.get('model', {})
    active_model = model_config.get('active', 'xgboost').lower()
    if active_model not in ("xgboost", "lightgbm", "catboost"):
        logger.warning("Unknown model.active in config; defaulting to xgboost")
        active_model = "xgboost"
    xgb_conf = config.get('xgboost', {})
    lgb_conf = config.get('lightgbm', {})
    catboost_conf = config.get('catboost', {})
    # Get monitoring directory from config, relative to project root
    monitoring_base_rel = config.get('paths', {}).get('monitoring_dir', 'output/monitoring')
    monitoring_base = PROJECT_ROOT / monitoring_base_rel
    monitoring_dir = monitoring_base / active_model
    
    logger.info("="*80)
    logger.info("Complete XGBoost Pipeline: Influencer/Observer Classification")
    logger.info("="*80)
    logger.info(f"Active model: {active_model}")
    
    # ========== Step 1: GPU Detection ==========
    logger.info("\n[Step 1] GPU Detection")
    logger.info("-" * 60)
    
    gpu_params = {}
    if active_model == "lightgbm":
        requested_gpu = args.use_gpu or lgb_conf.get('use_gpu', False)
        detected_gpu, detected_params = detect_gpu_for_lightgbm()
        gpu_available = requested_gpu
        gpu_params = detected_params if detected_gpu else {}
        if gpu_available:
            logger.info(f"LightGBM GPU requested; using params: {gpu_params or {'device': 'gpu'}}")
        else:
            logger.info("GPU disabled for LightGBM, using CPU mode")
    elif active_model == "catboost":
        requested_gpu = args.use_gpu or catboost_conf.get('use_gpu', False)
        detected_gpu, detected_params = detect_gpu_for_catboost()
        gpu_available = requested_gpu
        gpu_params = detected_params if detected_gpu else {}
        if gpu_available:
            logger.info(f"CatBoost GPU requested; using params: {gpu_params or {'task_type': 'GPU'}}")
        else:
            logger.info("GPU disabled for CatBoost, using CPU mode")
    else:  # active_model == "xgboost"
        if args.use_gpu:
            gpu_available, _, gpu_params = detect_gpu_for_xgboost()
            if not gpu_available:
                logger.warning("GPU requested but not available, falling back to CPU")
        else:
            gpu_available = False
            gpu_params = {'tree_method': 'hist'}
            logger.info("GPU disabled by user, using CPU mode")

    preprocess_dir = get_preprocessing_dir()
    preprocess_paths = _get_preprocessing_paths(preprocess_dir)
    cached_artifacts = _load_preprocessed_artifacts(preprocess_paths)
    preprocessing_loaded = cached_artifacts is not None
    if preprocessing_loaded:
        X_train = cached_artifacts["X_train"]
        X_test = cached_artifacts["X_test"]
        y_train = cached_artifacts["y_train"]
        test_ids = cached_artifacts["test_ids"]
        train_groups = cached_artifacts["train_groups"]
    else:
        X_train = X_test = y_train = test_ids = train_groups = None
    train_text_df = train_nontext_df = test_text_df = test_nontext_df = None
    
    # ========== Step 2: Data Loading ==========
    logger.info("\n[Step 2] Data Loading")
    logger.info("-" * 60)
    if preprocessing_loaded:
        logger.info("  Skipping data loading; using cached preprocessing outputs in %s", preprocess_dir)
    else:
        if args.kaggle_max:
            # Kaggle-max  baseline 
            logger.info(f"Loading training data via src.utils.data_processing: {args.train_file}")
            train_df = dp.load_data(args.train_file, clean_text=True)
            logger.info(f"  Training data shape: {train_df.shape}")
            
            logger.info(f"Loading test data via src.utils.data_processing: {args.test_file}")
            test_df = dp.load_data(args.test_file, clean_text=True)
            logger.info(f"  Test data shape: {test_df.shape}")
            
            train_text_df, train_nontext_df = dp.split_text_nontext(train_df.copy(), text_columns)
            test_text_df, test_nontext_df = dp.split_text_nontext(test_df.copy(), text_columns)
            
            if 'label' in train_df.columns:
                y_train = train_df['label'].copy()
                logger.info(f"  Training labels: {y_train.value_counts().to_dict()}")
            else:
                raise ValueError("Training data must contain 'label' column")
        else:
            #  JSONL loader
            logger.info(f"Loading training data: {args.train_file}")
            train_df = load_jsonl(args.train_file)
            logger.info(f"  Training data shape: {train_df.shape}")
            
            logger.info(f"Loading test data: {args.test_file}")
            test_df = load_jsonl(args.test_file)
            logger.info(f"  Test data shape: {test_df.shape}")
            
            train_text_df, train_nontext_df = dp.split_text_nontext(train_df.copy(), text_columns)
            test_text_df, test_nontext_df = dp.split_text_nontext(test_df.copy(), text_columns)
            
            # Extract labels and IDs
            if 'label' in train_df.columns:
                y_train = train_df['label'].copy()
                train_df = train_df.drop(columns=['label'])
                logger.info(f"  Training labels: {y_train.value_counts().to_dict()}")
            else:
                raise ValueError("Training data must contain 'label' column")
            
            if 'challenge_id' in test_df.columns:
                test_ids = test_df['challenge_id'].copy()
                test_df = test_df.drop(columns=['challenge_id'])
                logger.info(f"  Test IDs: {len(test_ids)} samples")
            else:
                raise ValueError("Test data must contain 'challenge_id' column")
    
    camembert_train_texts = None
    camembert_test_texts = None
    
    if not preprocessing_loaded:
        # Extract user IDs for GroupKFold BEFORE feature engineering removes user columns
        # ID GroupKFold
        train_groups = extract_user_id(train_df)

        # Prepare CamemBERT fused texts for reuse
        try:
            camembert_train_texts = extract_and_fuse_text_fields(train_df)
            camembert_test_texts = extract_and_fuse_text_fields(test_df)
        except Exception as exc:
            logger.warning("Failed to build CamemBERT fused texts: %s", exc)
    
    # ========== Step 3: CamemBERT Fine-tuning (if enabled) ==========
    # Read embedding parameters from config (needed for fine-tuning check)
    embedding_model = semantic_config.get('embedding_model', 'deberta')  # 'camembert' or 'deberta'
    use_finetuned_probabilities = semantic_config.get('use_finetuned_probabilities', False)
    finetuned_model_path = semantic_config.get('finetune_model_path', 'output/models/camembert_finetuned')
    
    # CamemBERT specific parameters (needed for fine-tuning)
    camembert_model_name = semantic_config.get('camembert_model_name', 'camembert-base')
    camembert_max_length = semantic_config.get('camembert_max_length', 256)
    camembert_batch_size = semantic_config.get('camembert_batch_size', 32)
    extract_style_features = semantic_config.get('extract_style_features', True)
    classify_urls = semantic_config.get('classify_urls', True)
    
    if preprocessing_loaded:
        logger.info("\n[Step 3] CamemBERT Fine-tuning (skipped, cached preprocessing detected)")
    elif use_finetuned_probabilities and embedding_model.lower() == 'camembert':
        logger.info("\n[Step 3] CamemBERT Fine-tuning Check")
        logger.info("-" * 60)
        
        try:
            from src.camembert.finetuning import train_camembert_classifier
            from src.camembert.finetune_camembert import load_finetuning_params, save_finetuning_params
        except (ImportError, OSError) as e:
            logger.error(f"  ✗ Failed to import CamemBERT modules: {e}")
            logger.warning(f"  Falling back to non-fine-tuned embeddings...")
            use_finetuned_probabilities = False
        else:
            finetune_model_path = Path(finetuned_model_path)
            finetune_params_file = finetune_model_path / 'finetuning_params.json'
            
            # Try to load existing fine-tuning parameters
            finetuning_params = load_finetuning_params(finetune_params_file)
            
            # Check if fine-tuned model exists
            model_exists = (
                finetune_model_path.exists() and 
                (finetune_model_path / 'config.json').exists() and
                ((finetune_model_path / 'pytorch_model.bin').exists() or 
                 (finetune_model_path / 'model.safetensors').exists())
            )
            
            if model_exists and finetuning_params:
                logger.info(f"  ✓ Fine-tuned model found at {finetune_model_path}")
                logger.info(f"  ✓ Fine-tuning parameters loaded from {finetune_params_file}")
                logger.info(f"    - Model: {finetuning_params.get('model_name', 'N/A')}")
                logger.info(f"    - Epochs: {finetuning_params.get('num_epochs', 'N/A')}")
                logger.info(f"    - Learning rate: {finetuning_params.get('learning_rate', 'N/A')}")
                logger.info(f"    - Batch size: {finetuning_params.get('batch_size', 'N/A')}")
                logger.info(f"    - Fine-tuned at: {finetuning_params.get('finetuned_at', 'N/A')}")
                logger.info(f"  Using existing fine-tuned model (skip fine-tuning)")
            else:
                # Need to fine-tune
                logger.info(f"  Fine-tuned model or parameters not found, starting fine-tuning...")
                
                # Get parameters from config or use defaults
                finetune_epochs = semantic_config.get('finetune_epochs', 3)
                finetune_learning_rate = semantic_config.get('finetune_learning_rate', 2e-5)
                finetune_batch_size = semantic_config.get('finetune_batch_size', 16)
                finetune_validation_split = semantic_config.get('finetune_validation_split', 0.2)
                finetune_early_stopping_patience = semantic_config.get('finetune_early_stopping_patience', 2)
                finetune_visualization_dir = semantic_config.get('finetune_visualization_dir', 'output/camembert/finetuning')
                finetune_save_plots = semantic_config.get('finetune_save_plots', True)
                
                logger.info(f"  Fine-tuning configuration:")
                logger.info(f"    - Model: {camembert_model_name}")
                logger.info(f"    - Epochs: {finetune_epochs}")
                logger.info(f"    - Learning rate: {finetune_learning_rate}")
                logger.info(f"    - Batch size: {finetune_batch_size}")
                logger.info(f"    - Max length: {camembert_max_length}")
                logger.info(f"    - Validation split: {finetune_validation_split}")
                logger.info(f"    - Early stopping patience: {finetune_early_stopping_patience}")
                
                try:
                    # Use preprocessed texts if available (unified preprocessing)
                    preprocessed_texts_for_finetune = None
                    if camembert_train_texts is not None:
                        # Apply same preprocessing pipeline: URL replacement + normalization
                        from src.preprocessing.camembert_preprocessing import (
                            classify_and_replace_urls, normalize_text_for_camembert
                        )
                        logger.info("  Applying unified preprocessing pipeline to texts...")
                        # camembert_train_texts is already fused, now apply URL replacement and normalization
                        texts_with_urls, _ = classify_and_replace_urls(train_df, camembert_train_texts)
                        preprocessed_texts_for_finetune = normalize_text_for_camembert(texts_with_urls)
                        logger.info("  ✓ Preprocessed texts ready (includes user.description)")
                    
                    model, tokenizer, training_history = train_camembert_classifier(
                        train_df,
                        model_name=camembert_model_name,
                        output_dir=finetune_model_path,
                        num_epochs=finetune_epochs,
                        learning_rate=finetune_learning_rate,
                        batch_size=finetune_batch_size,
                        max_length=camembert_max_length,
                        validation_split=finetune_validation_split,
                        early_stopping_patience=finetune_early_stopping_patience,
                        save_plots=finetune_save_plots,
                        visualization_dir=Path(finetune_visualization_dir),
                        preprocessed_texts=preprocessed_texts_for_finetune
                    )
                    logger.info(f"  ✓ Fine-tuning complete! Model saved to {finetune_model_path}")
                    
                    # Save fine-tuning parameters
                    finetuning_params = {
                        'model_name': camembert_model_name,
                        'output_dir': str(finetune_model_path),
                        'num_epochs': finetune_epochs,
                        'learning_rate': finetune_learning_rate,
                        'batch_size': finetune_batch_size,
                        'max_length': camembert_max_length,
                        'validation_split': finetune_validation_split,
                        'early_stopping_patience': finetune_early_stopping_patience,
                        'train_file': str(args.train_file),
                        'num_train_samples': len(train_df),
                        'final_metrics': training_history.get('overfitting_info', {}),
                        'model_exists': True
                    }
                    
                    save_finetuning_params(finetuning_params, finetune_params_file)
                    logger.info(f"  ✓ Fine-tuning parameters saved to {finetune_params_file}")
                    
                    # Log overfitting analysis
                    if training_history.get('overfitting_info', {}).get('overfitting_detected'):
                        logger.warning(f"  ⚠ Overfitting detected during fine-tuning")
                    elif training_history.get('overfitting_info', {}).get('underfitting_detected'):
                        logger.warning(f"  ⚠ Underfitting detected during fine-tuning")
                    else:
                        logger.info(f"  ✓ No overfitting/underfitting detected")
                    
                except Exception as e:
                    logger.error(f"  ✗ Fine-tuning failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.warning(f"  Falling back to non-fine-tuned embeddings...")
                    use_finetuned_probabilities = False
    
    # ========== Step 4: Feature Engineering ==========
    logger.info("\n[Step 4] Feature Engineering")
    logger.info("-" * 60)
    
    if preprocessing_loaded:
        logger.info("  Skipped feature engineering (using cached artifacts).")
    elif args.kaggle_max:
        # Kaggle-max  baseline 
        logger.warning(
            "  Kaggle-max mode enabled: using src.utils.data_processing engineer_features "
            "and keeping user-level IDs/names as categorical features for higher LB score "
            "(user-level leakage expected)."
        )
        
        # Train set: fit transformers (baseline pipeline)
        train_engineered, transformers = dp.engineer_features(train_df, is_train=True)
        # Test set: reuse transformers
        test_engineered, _ = dp.engineer_features(
            test_df,
            is_train=False,
            transformers=transformers,
        )
        
        # Prepare X, y, ids using shared utility
        X_train, y_train, _ = dp.prepare_xy(train_engineered, target_col='label', id_col='challenge_id')
        X_test, _, test_ids = dp.prepare_xy(test_engineered, target_col='label', id_col='challenge_id')
        
        # Ensure columns match as in baseline (add missing, drop extras, order match)
        common_cols = [c for c in X_train.columns if c in X_test.columns]
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        for c in missing_in_test:
            X_test[c] = 0  # Default fill as in baseline
        X_test = X_test[X_train.columns]
        
        logger.info(f"  Training features shape: {X_train.shape}")
        logger.info(f"  Test features shape: {X_test.shape}")
    else:
        logger.info("  Extracting metadata features (NO text content)")
        
        # Train set: fit transformers (with y_train for target encoding)
        # Read embedding parameters from config (embedding_model and CamemBERT params already defined above)
        use_text_embedding = semantic_config.get('use_semantic_features', True)
        text_embedding_n_components = semantic_config.get('pca_components', 32)
        
        logger.info(f"  Using embedding model: {embedding_model}")
        if embedding_model.lower() == 'camembert':
            logger.info(f"    - Model: {camembert_model_name}")
            logger.info(f"    - Max length: {camembert_max_length}")
            logger.info(f"    - Batch size: {camembert_batch_size}")
            logger.info(f"    - Style features: {extract_style_features}")
            logger.info(f"    - URL classification: {classify_urls}")
            logger.info(f"    - Use fine-tuned probabilities: {use_finetuned_probabilities}")
            if use_finetuned_probabilities:
                logger.info(f"    - Fine-tuned model path: {finetuned_model_path}")
        
        X_train, transformers = engineer_features(
            train_df,
            is_train=True,
            kaggle_max=False,
            y_train=y_train,  # Required for target encoding
            use_text_embedding=use_text_embedding,
            text_embedding_n_components=text_embedding_n_components,
            embedding_model=embedding_model,
            camembert_max_length=camembert_max_length,
            camembert_batch_size=camembert_batch_size,
            extract_style_features=extract_style_features,
            classify_urls=classify_urls,
            use_finetuned_probabilities=use_finetuned_probabilities,
            finetuned_model_path=finetuned_model_path if use_finetuned_probabilities else None,
        )
        logger.info(f"  Training features shape: {X_train.shape}")
        
        # Test set: use training transformers (NO refit)
        logger.info("  Applying feature engineering to test set (using training transformers)...")
        X_test, _ = engineer_features(
            test_df,
            is_train=False,
            transformers=transformers,
            kaggle_max=False,
            y_train=None,  # Not needed for test set
            use_text_embedding=use_text_embedding,
            text_embedding_n_components=text_embedding_n_components,
            embedding_model=embedding_model,
            camembert_max_length=camembert_max_length,
            camembert_batch_size=camembert_batch_size,
            extract_style_features=extract_style_features,
            classify_urls=classify_urls,
            use_finetuned_probabilities=use_finetuned_probabilities,
            finetuned_model_path=finetuned_model_path if use_finetuned_probabilities else None,
        )
        logger.info(f"  Test features shape: {X_test.shape}")
        
        # Ensure all features are numeric (safety check)
        non_numeric_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_test = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_train:
            logger.warning(f"  ⚠ Found {len(non_numeric_train)} non-numeric columns in training set, removing them")
            X_train = X_train.drop(columns=non_numeric_train)
        
        if non_numeric_test:
            logger.warning(f"  ⚠ Found {len(non_numeric_test)} non-numeric columns in test set, removing them")
            X_test = X_test.drop(columns=non_numeric_test)
        
        # Ensure feature columns match
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        if missing_in_test:
            logger.warning(f"  Adding {len(missing_in_test)} missing columns to test set (filled with NaN)")
            for col in missing_in_test:
                X_test[col] = np.nan
        
        if extra_in_test:
            logger.warning(f"  Removing {len(extra_in_test)} extra columns from test set")
            X_test = X_test.drop(columns=list(extra_in_test))
        
        # Ensure column order matches
        X_test = X_test[X_train.columns]
    
    if not preprocessing_loaded:
        _save_preprocessing_artifacts(
            paths=preprocess_paths,
            train_text_df=train_text_df,
            train_nontext_df=train_nontext_df,
            test_text_df=test_text_df,
            test_nontext_df=test_nontext_df,
            camembert_train_texts=camembert_train_texts,
            camembert_test_texts=camembert_test_texts,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            test_ids=test_ids,
            train_groups=train_groups,
            metadata={
                "text_columns": text_columns,
                "kaggle_max": args.kaggle_max,
                "preprocessing_version": "v1",
            },
        )
    
    # ========== Step 5: Extract User IDs ==========
    logger.info("\n[Step 5] Extract User IDs for GroupKFold")
    logger.info("-" * 60)
    
    # train_groups  train_df 
    logger.info(f"  Training: {train_groups.nunique()} unique users")
    
    # ========== Step 6: Hyperparameter Optimization ==========
    best_params = None
    mean_best_iterations = None
    optimization_result = None

    if active_model in ("lightgbm", "catboost") and args.optimize:
        logger.warning(f"Optuna optimization not implemented for {active_model} branch; skipping optimization.")
        args.optimize = False
    
    if args.optimize:
        logger.info("\n[Step 6] Hyperparameter Optimization (Optuna)")
        logger.info("-" * 60)
        
        optimization_result = optimize_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            groups=train_groups,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            use_gpu=gpu_available,
            gpu_params=gpu_params,
            timeout=args.timeout,
            verbose=True,
            use_kaggle_score=args.use_kaggle_score,
            X_test=X_test if args.use_kaggle_score else None,
            test_ids=test_ids if args.use_kaggle_score else None,
            competition=args.competition,
            cv_strategy=args.cv_strategy
        )
        
        best_params = optimization_result['best_params']
        mean_best_iterations = optimization_result['mean_best_iterations']
        
        score_type = "Kaggle Score" if args.use_kaggle_score else "CV Score"
        logger.info(f"\n  Best {score_type}: {optimization_result['best_score']:.4f}")
        logger.info(f"  Mean Best Iterations: {mean_best_iterations:.1f}")
    else:
        logger.info("\n[Step 6] Skipping Hyperparameter Optimization")
        logger.info("  (Use --optimize to enable)")
        
        # Use default parameters
        if args.kaggle_max:
            # Kaggle-max  xgboost_baseline  scale_pos_weight
            best_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
            }
            best_params.update(gpu_params)
            mean_best_iterations = 200
        else:
            if active_model == "lightgbm":
                best_params = lgb_conf.get('params', {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': ['binary_logloss', 'auc'],
                    'learning_rate': 0.05,
                    'num_leaves': 64,
                    'max_depth': -1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'min_child_samples': 20,
                    'n_estimators': 200,
                    'random_state': 42,
                    'n_jobs': -1,
                })
                mean_best_iterations = best_params.get('n_estimators', 200)
            elif active_model == "catboost":
                best_params = catboost_conf.get('params', {
                    'loss_function': 'Logloss',
                    'eval_metric': 'Logloss',
                    'learning_rate': 0.05,
                    'depth': 6,
                    'iterations': 200,
                    'subsample': 0.8,
                    'rsm': 0.8,
                    'l2_leaf_reg': 1.0,
                    'min_data_in_leaf': 3,
                    'random_seed': 42,
                    'thread_count': -1,
                    'verbose': False,
                })
                mean_best_iterations = best_params.get('iterations', 200)
            else:
                #  scale_pos_weight 
                negative_count = np.sum(y_train == 0)
                positive_count = np.sum(y_train == 1)
                scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.127
                
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,
                    'n_estimators': 200,
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'n_jobs': -1,
                }
                best_params.update(gpu_params)
                mean_best_iterations = 200
    
    # ========== Step 7: Cross-Validation ==========
    logger.info("\n[Step 7] Cross-Validation")
    logger.info("-" * 60)
    
    # Set n_estimators/iterations to 1.1x of mean best iterations from CV (if available)
    if args.kaggle_max and not args.optimize:
        # Kaggle-max + no optimization: keep exactly the configured n_estimators/iterations
        if active_model == "catboost":
            final_n_estimators = best_params.get('iterations', int(1.1 * mean_best_iterations))
        else:
            final_n_estimators = best_params.get('n_estimators', int(1.1 * mean_best_iterations))
    else:
        if active_model == "catboost":
            final_n_estimators = int(1.1 * mean_best_iterations) if mean_best_iterations else best_params.get('iterations', 200)
        else:
            final_n_estimators = int(1.1 * mean_best_iterations) if mean_best_iterations else best_params.get('n_estimators', 200)
    
    final_params = best_params.copy()
    if active_model == "catboost":
        final_params['iterations'] = final_n_estimators
        final_params = build_catboost_params(final_params, use_gpu=gpu_available, gpu_params=gpu_params if gpu_available else None)
    else:
        final_params['n_estimators'] = final_n_estimators
        if active_model == "lightgbm":
            final_params = build_lgbm_params(final_params, use_gpu=gpu_available, gpu_params=gpu_params if gpu_available else None)
    
    # Run Cross-Validation to get CV scores before final training
    logger.info(f"  Running {args.cv_folds}-Fold GroupKFold Cross-Validation...")
    if active_model == "catboost":
        logger.info(f"  iterations: {final_n_estimators}")
    else:
        logger.info(f"  n_estimators: {final_n_estimators}")
    lgbm_builder = None
    catboost_builder = None
    if active_model == "lightgbm":
        lgbm_builder = lambda params: create_lgbm_classifier(
            build_lgbm_params(params, use_gpu=gpu_available, gpu_params=gpu_params if gpu_available else None)
        )
    elif active_model == "catboost":
        catboost_builder = lambda params: create_catboost_classifier(
            build_catboost_params(params, use_gpu=gpu_available, gpu_params=gpu_params if gpu_available else None)
        )
    
    cv_scores, cv_best_iterations = group_kfold_cv(
        X=X_train,
        y=y_train,
        groups=train_groups,
        model_params=final_params,
        n_splits=args.cv_folds,
        use_gpu=gpu_available,
        gpu_params=gpu_params,
        model_type=active_model,
        model_builder=lgbm_builder if active_model == "lightgbm" else (catboost_builder if active_model == "catboost" else None)
    )
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    mean_cv_iterations = np.mean(cv_best_iterations) if cv_best_iterations else final_n_estimators
    
    logger.info(f"\n  Cross-Validation Results:")
    logger.info(f"    Mean Accuracy: {mean_cv_score:.4f} ({mean_cv_score*100:.2f}%)")
    logger.info(f"    Std Deviation: {std_cv_score:.4f} ({std_cv_score*100:.2f}%)")
    logger.info(f"    Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"    Mean Best Iterations: {mean_cv_iterations:.1f}")
    
    # Update n_estimators/iterations based on CV results if not from optimization
    if not args.optimize:
        final_n_estimators = int(1.1 * mean_cv_iterations)
        if active_model == "catboost":
            final_params['iterations'] = final_n_estimators
            logger.info(f"  Updated iterations to {final_n_estimators} (1.1x of CV mean best iterations)")
        else:
            final_params['n_estimators'] = final_n_estimators
            logger.info(f"  Updated n_estimators to {final_n_estimators} (1.1x of CV mean best iterations)")
    
    # ========== Step 8: Validation Monitoring & Plots ==========
    logger.info("\n[Step 8] Validation Monitoring & Plots")
    logger.info("-" * 60)
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split
    # Split into train, validation, and test sets for monitoring
    X_tr_monitor, X_temp, y_tr_monitor, y_temp = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )
    X_val_monitor, X_test_monitor, y_val_monitor, y_test_monitor = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    monitor_params = final_params.copy()
    test_metric_values = None
    if active_model == "xgboost":
        monitor_params.setdefault('early_stopping_rounds', 50)
        monitor_model = xgb.XGBClassifier(**monitor_params)
        monitor_model.fit(
            X_tr_monitor, y_tr_monitor,
            eval_set=[(X_tr_monitor, y_tr_monitor), (X_val_monitor, y_val_monitor), (X_test_monitor, y_test_monitor)],
            verbose=False
        )
        evals_result = monitor_model.evals_result()
        # Extract test set metrics from evals_result (usually validation_2)
        if 'validation_2' in evals_result:
            test_metric_values = evals_result['validation_2']
    elif active_model == "lightgbm":
        monitor_model = create_lgbm_classifier(monitor_params)
        fit_kwargs = {
            "eval_set": [
                (X_val_monitor, y_val_monitor),
                (X_tr_monitor, y_tr_monitor),
                (X_test_monitor, y_test_monitor),
            ],
            "eval_names": ["validation", "train", "test"],
            "eval_train_metric": True,
        }
        if "eval_metric" not in monitor_params:
            fit_kwargs["eval_metric"] = "binary_logloss"
        try:
            import lightgbm as lgb
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        except Exception:
            pass
        monitor_model.fit(X_tr_monitor, y_tr_monitor, **fit_kwargs)
        evals_result = getattr(monitor_model, "evals_result_", {}) or getattr(monitor_model, "evals_result", {})
        # Extract test set metrics from evals_result
        # LightGBM uses 0-indexed: valid_0 = first eval_set, valid_1 = second eval_set
        # Check both 'valid_X' and 'validation_X' formats
        test_keys = ['valid_1', 'validation_1', 'test', 'validation_2']
        test_metric_values = None
        for key in test_keys:
            if key in evals_result:
                test_metric_values = evals_result[key]
                break
    else:  # active_model == "catboost"
        monitor_model = create_catboost_classifier(monitor_params)
        # CatBoost uses eval_set parameter in fit()
        # GPU mode doesn't support multiple eval sets, use only validation set
        if monitor_params.get('task_type') == 'GPU':
            # GPU mode: only use validation set
            monitor_model.fit(
                X_tr_monitor, y_tr_monitor,
                eval_set=[(X_val_monitor, y_val_monitor)],
                verbose=False,
                early_stopping_rounds=50
            )
            # For GPU mode, use validation set metrics as test metrics
            evals_result = getattr(monitor_model, "evals_result_", {}) or {}
            test_metric_values = evals_result.get('validation', {}) or evals_result.get('validation_0', {})
        else:
            # CPU mode: can use multiple eval sets
            monitor_model.fit(
                X_tr_monitor, y_tr_monitor,
                eval_set=[(X_val_monitor, y_val_monitor), (X_test_monitor, y_test_monitor)],
                verbose=False,
                early_stopping_rounds=50
            )
            # CatBoost stores evaluation results in evals_result_ attribute
            evals_result = getattr(monitor_model, "evals_result_", {}) or {}
            # CatBoost uses 'validation' prefix for eval sets
            test_keys = ['validation_1', 'validation', 'test']
            test_metric_values = None
            for key in test_keys:
                if key in evals_result:
                    test_metric_values = evals_result[key]
                    break

    y_val_pred = monitor_model.predict(X_val_monitor)
    if hasattr(monitor_model, "predict_proba"):
        y_val_proba = monitor_model.predict_proba(X_val_monitor)[:, 1]
    else:
        y_val_proba = np.zeros_like(y_val_pred, dtype=float)

    generate_training_report(
        evals_result=evals_result,
        y_true=y_val_monitor,
        y_pred=y_val_pred,
        y_pred_proba=y_val_proba,
        output_dir=monitoring_dir,
        test_metric_values=test_metric_values
    )
    # Also emit an aggregated boosting summary plot (one color per model, train vs val)
    try:
        _save_and_plot_boosting_summary(
            evals_result=evals_result,
            monitoring_base=monitoring_base,
            report_dir=PROJECT_ROOT / "report",
            model_name=active_model,
        )
    except Exception as agg_exc:
        logger.warning("Failed to generate boosting summary plot: %s", agg_exc)

    log_error_statistics(
        y_true=y_val_monitor,
        y_pred=y_val_pred,
        ids=X_val_monitor.index.astype(str),
        output_dir=monitoring_dir
    )
    
    # ========== Step 9: Final Training ==========
    logger.info("\n[Step 9] Final Training on Full Dataset")
    logger.info("-" * 60)
    
    logger.info("  Training final model on full training set...")
    if active_model == "lightgbm":
        final_model = create_lgbm_classifier(final_params)
    elif active_model == "catboost":
        final_model = create_catboost_classifier(final_params)
    else:
        final_model = xgb.XGBClassifier(**final_params)
    
    # Convert to numpy arrays
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    final_model.fit(X_train_array, y_train_array)
    logger.info("  ✓ Final model trained")
    
    # Save model and metadata
    logger.info("\n  Saving model and metadata...")
    from src.utils.paths import get_models_output_dir
    models_dir = get_models_output_dir()
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if active_model == "lightgbm":
        model_prefix = "lightgbm_model"
    elif active_model == "catboost":
        model_prefix = "catboost_model"
    else:
        model_prefix = "xgboost_model"
    model_filename = f"{model_prefix}_{timestamp}.pkl"
    model_path = models_dir / model_filename
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    logger.info(f"  ✓ Model saved: {model_path}")
    
    # Save metadata (feature names, parameters, etc.)
    metadata = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'feature_names': X_train.columns.tolist(),
        'n_features': len(X_train.columns),
        'n_samples': len(X_train),
        'model_params': final_params,
        'model_type': active_model,
        'cv_strategy': args.cv_strategy,
        'optimized': args.optimize,
        'best_score': optimization_result['best_score'] if args.optimize and optimization_result else None,
        'mean_best_iterations': mean_best_iterations,
        'n_estimators': final_n_estimators if active_model != "catboost" else None,
        'iterations': final_n_estimators if active_model == "catboost" else None,
        'cv_scores': cv_scores if 'cv_scores' in locals() else None,
        'mean_cv_score': float(mean_cv_score) if 'mean_cv_score' in locals() else None,
        'std_cv_score': float(std_cv_score) if 'std_cv_score' in locals() else None,
        'mean_cv_iterations': float(mean_cv_iterations) if 'mean_cv_iterations' in locals() else None
    }
    
    metadata_filename = f"{model_prefix}_{timestamp}_metadata.json"
    metadata_path = models_dir / metadata_filename
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Metadata saved: {metadata_path}")
    
    # Also save latest model (for easy access)
    latest_model_path = models_dir / f"{model_prefix}_latest.pkl"
    latest_metadata_path = models_dir / f"{model_prefix}_latest_metadata.json"
    
    import shutil
    shutil.copy2(model_path, latest_model_path)
    shutil.copy2(metadata_path, latest_metadata_path)
    logger.info(f"  ✓ Latest model symlinks created")
    
    # ========== Step 10: SHAP Analysis ==========
    if args.shap and active_model in ("xgboost", "lightgbm"):
        logger.info("\n[Step 10] SHAP Analysis")
        logger.info("-" * 60)
        
        # Use a subset of training data for SHAP (for speed)
        shap_sample_size = min(args.shap_samples, len(X_train))
        shap_indices = np.random.choice(len(X_train), shap_sample_size, replace=False)
        X_shap = X_train.iloc[shap_indices]
        
        try:
            # Import SHAP analysis function (lazy import to handle import errors gracefully)
            from src.xgboost.solution.shap_analysis import analyze_with_shap
            
            # Predict on training set for error analysis
            logger.info("  Predicting on training set for error analysis...")
            y_train_pred = final_model.predict(X_train.values)
            y_train_true = y_train.values
            
            # Get labels for SHAP samples
            y_shap_true = y_train_true[shap_indices]
            y_shap_pred = y_train_pred[shap_indices]
            
            # Create a validation set using GroupKFold for error analysis
            logger.info("  Creating validation set using GroupKFold...")
            from src._imports import sklearn_GroupKFold as GroupKFold
            
            # Use a single fold for validation set
            gkf = GroupKFold(n_splits=5)
            X_train_array = X_train.values
            y_train_array = y_train.values
            train_groups_array = train_groups.values
            
            # Get first fold as validation set
            train_idx, val_idx = next(gkf.split(X_train_array, y_train_array, groups=train_groups_array))
            X_val_shap = X_train.iloc[val_idx]
            y_val_true = y_train_array[val_idx]
            
            # Predict on validation set
            logger.info("  Predicting on validation set...")
            y_val_pred = final_model.predict(X_val_shap.values)
            
            # Perform SHAP analysis with error analysis
            shap_result = analyze_with_shap(
                model=final_model,
                X_val=X_shap,
                feature_names=X_train.columns.tolist(),
                max_samples=args.shap_samples,
                max_display=20,
                output_dir=args.shap_output_dir,
                plot=True,
                generate_waterfall=True,
                generate_force=True,
                generate_dependence=True,
                generate_interaction=False,  # Disable by default (slower)
                generate_interaction_dependence=True,  # Enable interaction dependence plots
                num_waterfall_samples=5,
                num_dependence_features=10,
                num_interaction_dependence_features=10,
                # Error analysis parameters
                y_true=y_shap_true,
                y_pred=y_shap_pred,
                analyze_errors=True,
                max_error_force_plots=10
            )
            logger.info("  ✓ SHAP analysis complete")
            
            # Additional error analysis for validation set
            logger.info("\n  Performing error analysis on validation set...")
            try:
                from src.xgboost.solution.shap_analysis import analyze_misclassified_samples
                from src.utils.paths import get_shap_dir
                
                val_error_output_dir = str(get_shap_dir() / 'errors' / 'validation')
                
                val_error_result = analyze_misclassified_samples(
                    model=final_model,
                    X=X_val_shap,
                    y_true=y_val_true,
                    y_pred=y_val_pred,
                    feature_names=X_train.columns.tolist(),
                    output_dir=val_error_output_dir,
                    max_error_force_plots=10,
                    max_display=20
                )
                logger.info("  ✓ Validation set error analysis complete")
            except Exception as e:
                logger.warning(f"  ⚠ Validation set error analysis failed: {e}")
                import traceback
                logger.debug(f"  Full error: {traceback.format_exc()}")
            
        except ImportError as e:
            logger.warning(f"  ⚠ SHAP analysis skipped: {e}")
            logger.info("  Tip: Install SHAP with: pip install shap")
            logger.info("  Or ensure SHAP is available in /Data/yuguang.yao/pytorch_packages")
        except Exception as e:
            logger.warning(f"  ⚠ SHAP analysis failed: {e}")
            import traceback
            logger.debug(f"  Full error: {traceback.format_exc()}")
    else:
        logger.info("\n[Step 10] Skipping SHAP Analysis")
    
    # ========== Step 11: Test Set Prediction ==========
    logger.info("\n[Step 11] Test Set Prediction")
    logger.info("-" * 60)
    
    if not args.kaggle_max:
        #  NaN /
        # Handle missing values (XGBoost can handle NaN, but ensure consistency)
        # Fill NaN with median for numeric columns (only if column has valid values)
        for col in X_test.columns:
            if X_test[col].dtype in [np.float64, np.int64]:
                # Check if column has any non-null values in training set
                train_non_null = X_train[col].notna().sum()
                if train_non_null > 0:
                    median_val = X_train[col].median()
                    if pd.notna(median_val):
                        X_test[col] = X_test[col].fillna(median_val)
                    else:
                        # If median is NaN, fill with 0
                        X_test[col] = X_test[col].fillna(0)
                else:
                    # If all values are NaN in training, fill with 0
                    X_test[col] = X_test[col].fillna(0)
    
    logger.info("  Generating predictions...")
    
    # Convert to float32 for better performance and GPU compatibility
    # XGBoost will handle device mismatch automatically (CPU data with GPU model)
    # The warning is harmless - XGBoost will use DMatrix conversion internally
    X_test_array = X_test.values.astype(np.float32)
    predictions = final_model.predict(X_test_array)
    logger.info(f"  Predictions: {np.bincount(predictions)}")
    
    # ========== Step 11: Save Submission ==========
    logger.info("\n[Step 12] Save Submission")
    logger.info("-" * 60)
    
    save_submission(predictions, test_ids, args.output)
    
    # ========== Step 13: Kaggle Submission (Optional) ==========
    if args.submit:
        logger.info("\n[Step 13] Kaggle Submission")
        logger.info("-" * 60)
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            import time
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            logger.info("  ✓ Kaggle API authenticated")
            
            # Submit file
            logger.info(f"  Submitting {args.output} to competition: {args.competition}")
            logger.info(f"  Message: {args.submission_message}")
            
            api.competition_submit(
                file_name=args.output,
                message=args.submission_message,
                competition=args.competition
            )
            logger.info("  ✓ Submission uploaded successfully")
            
            # Wait a bit for submission to process
            logger.info("  Waiting for submission to process...")
            time.sleep(5)
            
            # Get submission status and score
            submissions = api.competition_submissions(args.competition)
            if submissions:
                latest_submission = submissions[0]  # Most recent submission
                logger.info(f"\n  {'='*60}")
                logger.info(f"  Submission Status:")
                
                # ApiSubmission is an object, not a dict - use attribute access
                try:
                    submission_date = getattr(latest_submission, 'date', None) or getattr(latest_submission, 'submittedAt', 'N/A')
                    submission_desc = getattr(latest_submission, 'description', None) or getattr(latest_submission, 'fileName', 'N/A')
                    submission_status = getattr(latest_submission, 'status', None) or getattr(latest_submission, 'status', 'N/A')
                    
                    logger.info(f"    Date: {submission_date}")
                    logger.info(f"    Description: {submission_desc}")
                    logger.info(f"    Status: {submission_status}")
                    
                    # Try to get score (different attribute names in different API versions)
                    score = None
                    for attr_name in ['publicScore', 'score', 'publicScoreFloat', 'scoreFloat']:
                        score = getattr(latest_submission, attr_name, None)
                        if score is not None:
                            break
                    
                    if score is not None:
                        logger.info(f"    Score: {score}")
                        logger.info(f"  {'='*60}")
                    else:
                        logger.info(f"    Score: Processing... (check Kaggle website)")
                        logger.info(f"  {'='*60}")
                        logger.info(f"  Note: Score may take a few minutes to appear")
                        logger.info(f"  Check status at: https://www.kaggle.com/competitions/{args.competition}/submissions")
                except Exception as e:
                    logger.warning(f"  Could not parse submission details: {e}")
                    logger.info(f"  Submission uploaded successfully. Check status at:")
                    logger.info(f"  https://www.kaggle.com/competitions/{args.competition}/submissions")
            else:
                logger.warning("  Could not retrieve submission status")
                logger.info(f"  Submission uploaded successfully. Check status at:")
                logger.info(f"  https://www.kaggle.com/competitions/{args.competition}/submissions")
                
        except ImportError:
            logger.error("  ✗ Kaggle API not installed")
            logger.error("  Install with: pip install kaggle")
            logger.error("  Or submit manually: kaggle competitions submit -c {} -f {} -m \"{}\"".format(
                args.competition, args.output, args.submission_message
            ))
        except Exception as e:
            logger.error(f"  ✗ Kaggle submission failed: {e}")
            logger.error("  You can submit manually with:")
            logger.error("  kaggle competitions submit -c {} -f {} -m \"{}\"".format(
                args.competition, args.output, args.submission_message
            ))
    
    # ========== Summary ==========
    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info("="*80)
    logger.info(f"  Submission file: {args.output}")
    logger.info(f"  Active model: {active_model}")
    logger.info(f"  GPU acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    if args.optimize:
        logger.info(f"  Best CV Score: {optimization_result['best_score']:.4f}")
    if args.submit:
        logger.info(f"  Kaggle submission: {'Completed' if args.submit else 'Skipped'}")
    logger.info("="*80)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Main Pipeline Script
主流程脚本 / Script de pipeline principal / Main Pipeline Script / Haupt-Pipeline-Skript

Complete end-to-end pipeline for Kaggle "Influencer or Observer" classification.
完整的端到端流程，用于Kaggle "Influencer or Observer" 分类挑战 / Complete end-to-end pipeline for Kaggle "Influencer or Observer" classification challenge

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
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional

# 添加 /Data 目录中的包路径（如果存在）
# Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

# Add solution directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Add parent directory (src/xgboost) to path for utils and feature_engineering
xgboost_dir = os.path.dirname(script_dir)
if xgboost_dir not in sys.path:
    sys.path.insert(0, xgboost_dir)

# Add project root to path to allow imports from src (for baseline-style pipeline)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import solution modules
from utils import (
    detect_gpu_for_xgboost,
    load_jsonl,
    extract_user_id,
    save_submission,
    ensure_dir
)
from validation import group_kfold_cv
from hyperparameter_optimization import optimize_hyperparameters

# Centralized preprocessing/feature engineering
from src.preprocessing.xgboost_feature_engineering import engineer_features

# Import shared baseline-style data processing utilities
from src.preprocessing import data_processing as dp
from src.utils.config import load_config
from src.utils.paths import get_output_dir
from src.vis_stat.training_monitor import generate_training_report
# SHAP analysis imported lazily (only when needed) to handle import errors gracefully

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    
    logger.info("="*80)
    logger.info("Complete XGBoost Pipeline: Influencer/Observer Classification")
    logger.info("="*80)
    
    # ========== Step 1: GPU Detection ==========
    logger.info("\n[Step 1] GPU Detection")
    logger.info("-" * 60)
    
    if args.use_gpu:
        gpu_available, tree_method, gpu_params = detect_gpu_for_xgboost()
        if not gpu_available:
            logger.warning("GPU requested but not available, falling back to CPU")
    else:
        gpu_available = False
        tree_method = 'hist'
        gpu_params = {'tree_method': 'hist'}
        logger.info("GPU disabled by user, using CPU mode")
    
    # ========== Step 2: Data Loading ==========
    logger.info("\n[Step 2] Data Loading")
    logger.info("-" * 60)
    
    if args.kaggle_max:
        # Kaggle-max 模式：使用与 baseline 相同的加载与展开逻辑
        logger.info(f"Loading training data via src.utils.data_processing: {args.train_file}")
        train_df = dp.load_data(args.train_file, clean_text=True)
        logger.info(f"  Training data shape: {train_df.shape}")
        
        logger.info(f"Loading test data via src.utils.data_processing: {args.test_file}")
        test_df = dp.load_data(args.test_file, clean_text=True)
        logger.info(f"  Test data shape: {test_df.shape}")
        
        if 'label' in train_df.columns:
            y_train = train_df['label'].copy()
            logger.info(f"  Training labels: {y_train.value_counts().to_dict()}")
        else:
            raise ValueError("Training data must contain 'label' column")
    else:
        # 严格模式：使用本地 JSONL loader
        logger.info(f"Loading training data: {args.train_file}")
        train_df = load_jsonl(args.train_file)
        logger.info(f"  Training data shape: {train_df.shape}")
        
        logger.info(f"Loading test data: {args.test_file}")
        test_df = load_jsonl(args.test_file)
        logger.info(f"  Test data shape: {test_df.shape}")
        
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
    
    # Extract user IDs for GroupKFold BEFORE feature engineering removes user columns
    # 在特征工程删除用户字段之前先提取用户ID，用于 GroupKFold，避免退化为按样本索引分组
    train_groups = extract_user_id(train_df)
    
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
    
    if use_finetuned_probabilities and embedding_model.lower() == 'camembert':
        logger.info("\n[Step 3] CamemBERT Fine-tuning Check")
        logger.info("-" * 60)
        
        try:
            from pathlib import Path
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
                        visualization_dir=Path(finetune_visualization_dir)
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
    
    if args.kaggle_max:
        # Kaggle-max 模式：完全复用 baseline 的特征工程与列对齐逻辑
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
    
    # ========== Step 5: Extract User IDs ==========
    logger.info("\n[Step 5] Extract User IDs for GroupKFold")
    logger.info("-" * 60)
    
    # train_groups 已在特征工程之前从原始 train_df 中提取，确保真正按用户分组
    logger.info(f"  Training: {train_groups.nunique()} unique users")
    
    # ========== Step 6: Hyperparameter Optimization ==========
    best_params = None
    mean_best_iterations = None
    optimization_result = None
    
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
            # Kaggle-max 模式：尽量对齐 xgboost_baseline 的默认参数（不使用 scale_pos_weight）
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
            # 严格模式：保留原有带 scale_pos_weight 的设置
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
    
    # Set n_estimators to 1.1x of mean best iterations from CV (if available)
    if args.kaggle_max and not args.optimize:
        # Kaggle-max + no optimization: keep exactly the configured n_estimators
        final_n_estimators = best_params.get('n_estimators', int(1.1 * mean_best_iterations))
    else:
        final_n_estimators = int(1.1 * mean_best_iterations) if mean_best_iterations else best_params.get('n_estimators', 200)
    
    final_params = best_params.copy()
    final_params['n_estimators'] = final_n_estimators
    
    # Run Cross-Validation to get CV scores before final training
    logger.info(f"  Running {args.cv_folds}-Fold GroupKFold Cross-Validation...")
    logger.info(f"  n_estimators: {final_n_estimators}")
    
    cv_scores, cv_best_iterations = group_kfold_cv(
        X=X_train,
        y=y_train,
        groups=train_groups,
        model_params=final_params,
        n_splits=args.cv_folds,
        use_gpu=gpu_available,
        gpu_params=gpu_params
    )
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    mean_cv_iterations = np.mean(cv_best_iterations) if cv_best_iterations else final_n_estimators
    
    logger.info(f"\n  Cross-Validation Results:")
    logger.info(f"    Mean Accuracy: {mean_cv_score:.4f} ({mean_cv_score*100:.2f}%)")
    logger.info(f"    Std Deviation: {std_cv_score:.4f} ({std_cv_score*100:.2f}%)")
    logger.info(f"    Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    logger.info(f"    Mean Best Iterations: {mean_cv_iterations:.1f}")
    
    # Update n_estimators based on CV results if not from optimization
    if not args.optimize:
        final_n_estimators = int(1.1 * mean_cv_iterations)
        final_params['n_estimators'] = final_n_estimators
        logger.info(f"  Updated n_estimators to {final_n_estimators} (1.1x of CV mean best iterations)")
    
    # ========== Step 8: Final Training ==========
    logger.info("\n[Step 8] Final Training on Full Dataset")
    logger.info("-" * 60)
    
    logger.info("  Training final model on full training set...")
    final_model = xgb.XGBClassifier(**final_params)
    
    # Convert to numpy arrays
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    final_model.fit(
        X_train_array,
        y_train_array,
        eval_set=[(X_train_array, y_train_array)],
        verbose=False
    )
    logger.info("  ✓ Final model trained")
    
    # Save model and metadata
    logger.info("\n  Saving model and metadata...")
    from src.utils.paths import get_models_output_dir
    models_dir = get_models_output_dir()
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_model_{timestamp}.pkl"
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
        'cv_strategy': args.cv_strategy,
        'optimized': args.optimize,
        'best_score': optimization_result['best_score'] if args.optimize and optimization_result else None,
        'mean_best_iterations': mean_best_iterations,
        'n_estimators': final_n_estimators,
        'cv_scores': cv_scores if 'cv_scores' in locals() else None,
        'mean_cv_score': float(mean_cv_score) if 'mean_cv_score' in locals() else None,
        'std_cv_score': float(std_cv_score) if 'std_cv_score' in locals() else None,
        'mean_cv_iterations': float(mean_cv_iterations) if 'mean_cv_iterations' in locals() else None
    }
    
    metadata_filename = f"xgboost_model_{timestamp}_metadata.json"
    metadata_path = models_dir / metadata_filename
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Metadata saved: {metadata_path}")
    
    # Also save latest model (for easy access)
    latest_model_path = models_dir / "xgboost_model_latest.pkl"
    latest_metadata_path = models_dir / "xgboost_model_latest_metadata.json"
    
    import shutil
    shutil.copy2(model_path, latest_model_path)
    shutil.copy2(metadata_path, latest_metadata_path)
    logger.info(f"  ✓ Latest model symlinks created")

    # Generate training monitoring visuals/statistics
    try:
        logger.info("\n  Generating training monitoring visuals/statistics...")
        monitor_dir = get_output_dir('monitoring')
        evals_result = getattr(final_model, 'evals_result', lambda: {})()
        # If evals_result is empty (e.g., older xgboost), skip plotting
        if evals_result:
            y_train_pred_proba = final_model.predict_proba(X_train_array)[:, 1]
            y_train_pred = final_model.predict(X_train_array)
            generate_training_report(
                evals_result=evals_result,
                y_true=y_train_array,
                y_pred=y_train_pred,
                y_pred_proba=y_train_pred_proba,
                output_dir=monitor_dir
            )
            logger.info(f"  ✓ Training monitoring saved to {monitor_dir}")
        else:
            logger.info("  No evals_result available; skipping training curves.")
    except Exception as e:
        logger.warning(f"  ⚠ Training monitoring generation failed: {e}")
    
    # ========== Step 9: SHAP Analysis ==========
    if args.shap:
        logger.info("\n[Step 9] SHAP Analysis")
        logger.info("-" * 60)
        
        # Use a subset of training data for SHAP (for speed)
        shap_sample_size = min(args.shap_samples, len(X_train))
        shap_indices = np.random.choice(len(X_train), shap_sample_size, replace=False)
        X_shap = X_train.iloc[shap_indices]
        
        try:
            # Import SHAP analysis function (lazy import to handle import errors gracefully)
            from shap_analysis import analyze_with_shap
            
            # Predict on training set for error analysis
            logger.info("  Predicting on training set for error analysis...")
            y_train_pred = final_model.predict(X_train.values)
            y_train_true = y_train.values
            
            # Get labels for SHAP samples
            y_shap_true = y_train_true[shap_indices]
            y_shap_pred = y_train_pred[shap_indices]
            
            # Create a validation set using GroupKFold for error analysis
            logger.info("  Creating validation set using GroupKFold...")
            from sklearn.model_selection import GroupKFold
            
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
                from shap_analysis import analyze_misclassified_samples
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
        logger.info("\n[Step 9] Skipping SHAP Analysis")
    
    # ========== Step 10: Test Set Prediction ==========
    logger.info("\n[Step 10] Test Set Prediction")
    logger.info("-" * 60)
    
    if not args.kaggle_max:
        # 严格模式：保持原有的 NaN 填充逻辑，避免训练/推理缺失模式不一致
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
    
    # ========== Step 10: Save Submission ==========
    logger.info("\n[Step 11] Save Submission")
    logger.info("-" * 60)
    
    save_submission(predictions, test_ids, args.output)
    
    # ========== Step 10: Kaggle Submission (Optional) ==========
    if args.submit:
        logger.info("\n[Step 10] Kaggle Submission")
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
    logger.info(f"  GPU acceleration: {'Enabled' if gpu_available else 'Disabled'}")
    if args.optimize:
        logger.info(f"  Best CV Score: {optimization_result['best_score']:.4f}")
    if args.submit:
        logger.info(f"  Kaggle submission: {'Completed' if args.submit else 'Skipped'}")
    logger.info("="*80)


if __name__ == '__main__':
    main()


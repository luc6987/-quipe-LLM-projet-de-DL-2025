#!/usr/bin/env python3
"""
Main Pipeline Script
 (Refactored)

Integrates:
1. Configuration Centralization (setup.yml)
2. Part I: Data Preprocessing (Libraries, Semantic, Text Split)
3. Part II: Iterative Viz & Optimization (SHAP, Auto-Pruning)
4. Part III: Monitoring & Error Analysis (Training Curves, Error Stats)
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
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import load_config
from src.preprocessing import data_processing as dp
from src.preprocessing.library_builder import build_libraries
from src.preprocessing.data_processing import filter_features_by_shap
from src.vis_stat.error_analysis import log_error_statistics
from src.vis_stat.training_monitor import generate_training_report
from src.vis_stat.plan_optimizer import optimize_plan_by_shap
from src.xgboost.utils import (
    detect_gpu_for_xgboost,
    load_jsonl,
    extract_user_id,
    save_submission,
    ensure_dir
)
from src.xgboost.validation import group_kfold_cv
# Lazy import SHAP
from src.vis_stat.shap_analysis import analyze_with_shap

# Import feature interaction analysis
try:
    from src.vis_stat.feature_interaction_analysis import analyze_feature_interactions
    INTERACTION_ANALYSIS_AVAILABLE = True
except ImportError:
    INTERACTION_ANALYSIS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConfigObject:
    """Helper to access dict as attributes"""
    def __init__(self, dictionary):
        self._d = dictionary
        for k, v in dictionary.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigObject(v))
            else:
                setattr(self, k, v)
    
    def get(self, key, default=None):
        return self._d.get(key, default)

def main():
    # 1. Load Configuration (Single Source of Truth)
    config = load_config()
    xg_conf = config.get('xgboost', {})
    prep_conf = config.get('preprocessing', {})
    sem_conf = config.get('semantic', {})
    mon_conf = config.get('monitoring', {})
    opt_conf = config.get('optimization', {})
    
    # Paths
    output_dir = Path(config['paths']['output_dir'])
    viz_dir = Path(config['paths']['visualizations_dir'])
    shap_dir = Path(config['paths']['shap_dir'])
    mon_dir = Path(config['paths']['monitoring_dir']) / "xgboost"
    plan_path = project_root / "src/xgboost/feature_processing_plan.json"
    
    logger.info("="*80)
    logger.info("Refactored XGBoost Pipeline (Strict Rules Compliant)")
    logger.info("="*80)
    
    # 2. Preprocessing Phase
    logger.info("\n[Phase 1] Data Preprocessing")
    
    # Load raw data
    train_file = project_root / xg_conf['train_file']
    test_file = project_root / xg_conf['test_file']
    
    # --- Split Text/Non-Text & Build Libraries (Train Only) ---
    logger.info("  Loading raw training data...")
    df_train_raw = dp.load_data(str(train_file), clean_text=False) # We will clean text if needed later
    
    train_text, train_non_text = dp.split_text_nontext(
        df_train_raw, prep_conf.get('text_columns', [])
    )
    
    # Build Libraries
    logger.info("  Building Hashtag/Mention Libraries...")
    # Extract text from main column for library building
    raw_texts = train_text['text'].dropna().tolist() if 'text' in train_text.columns else []
    libs = build_libraries(
        raw_texts, 
        top_k_hashtags=prep_conf.get('hashtag_library_top_k', 1000),
        top_k_mentions=prep_conf.get('mention_library_top_k', 1000),
        output_dir=viz_dir
    )
    
    # Clean up large text list
    del raw_texts
    
    # --- Feature Engineering (Train) ---
    logger.info("  Engineering features (Train)...")
    df_train_eng, transformers = dp.engineer_features(
        train_non_text,
        config=config,
        is_train=True,
        transformers=None,
        hashtag_lib=libs['hashtag_lib'],
        mention_lib=libs['mention_lib']
    )
    
    # --- Semantic Merge (Train) ---
    if sem_conf.get('use_semantic_features', False):
        logger.info("  Merging Semantic Features (Train)...")
        df_train_eng = dp.merge_semantic_features(
            df_train_eng, 
            project_root / sem_conf['embedding_file'],
            sem_conf.get('embedding_dim', 768)
        )

    # --- Prepare Train XY ---
    y_train = df_train_eng['label']
    X_train = df_train_eng.drop(columns=['label'])
    
    # Extract IDs for Error Analysis
    if 'id_str' in df_train_raw.columns:
        train_ids = df_train_raw['id_str'] # Valid twitter ID
    elif 'id' in df_train_raw.columns:
         train_ids = df_train_raw['id']
    else:
        train_ids = pd.Series(range(len(df_train_eng))).astype(str)

    # Extract User IDs for GroupKFold (BEFORE feature engineering removes user columns)
    # ID GroupKFold
    logger.info("  Extracting user IDs for GroupKFold...")
    try:
        user_groups = extract_user_id(df_train_raw)
    except Exception as e:
        logger.warning(f"  Could not extract user IDs: {e}. Using index as fallback.")
        user_groups = pd.Series(range(len(df_train_raw)), index=df_train_raw.index)
        # Align with df_train_eng if lengths differ
        if len(user_groups) != len(df_train_eng):
            user_groups = pd.Series(range(len(df_train_eng)))

    # 3. Model Training & Monitoring
    logger.info("\n[Phase 2] Training & Monitoring")
    
    # Run Cross-Validation to get CV scores
    logger.info("  Running 5-Fold GroupKFold Cross-Validation...")
    params = xg_conf['params'].copy()
    gpu_params = None
    use_gpu = xg_conf.get('use_gpu', False)
    if use_gpu:
         avail, method, gparams = detect_gpu_for_xgboost()
         if avail:
             params.update(gparams)
             gpu_params = gparams
    
    cv_scores, best_iterations = group_kfold_cv(
        X=X_train,
        y=y_train,
        groups=user_groups,
        model_params=params,
        n_splits=5,
        use_gpu=use_gpu,
        gpu_params=gpu_params
    )
    
    # Store CV scores for later printing
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    # Validation Split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(
        X_train, y_train, train_ids, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
    
    logger.info("  Training XGBoost with Early Stopping...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )
    
    # Validation Predictions (Probabilities & Classes)
    logger.info("  Generating Validation Predictions...")
    val_preds = model.predict(X_val)
    # Probabilities for AUC (class 1)
    val_probs = model.predict_proba(X_val)[:, 1]

    # --- Comprehensive Monitoring Report ---
    if mon_conf.get('plot_training_curve', True):
        logger.info(f"  Generating Training Monitor Report (Curves, CM, ROC)...")
        results = model.evals_result()
        generate_training_report(
            evals_result=results,
            y_true=y_val,
            y_pred=val_preds,
            y_pred_proba=val_probs,
            output_dir=mon_dir
        )

    # --- Error Analysis (Targeted) ---
    logger.info("  Running Targeted Error Analysis...")
    if mon_conf.get('save_error_stats', True):
        log_error_statistics(
            y_true=y_val, 
            y_pred=val_preds, 
            ids=id_val, 
            output_dir=mon_dir
        )

    # 4. Iterative Visualization & Optimization
    logger.info("\n[Phase 3] SHAP Analysis & Optimization")
    
    if xg_conf.get('shap', True):
        shap_res = analyze_with_shap(
            model=model,
            X_val=X_val,
            feature_names=X_train.columns.tolist(),
            output_dir=str(shap_dir),
            max_samples=xg_conf.get('shap_samples', 1000),
            min_shap_threshold=opt_conf.get('min_shap_threshold', 0.001)
        )
        
        # --- Auto Optimization ---
        logger.info("  Running Plan Optimizer (Pruning)...")
        pruned_count = optimize_plan_by_shap(
            plan_path=plan_path,
            importance_csv_path=shap_dir / 'feature_importance.csv',
            drop_threshold=opt_conf.get('shap_drop_threshold', 0.001),
            backup_dir=Path(config['paths']['feature_plans_dir'])
        )
        if pruned_count > 0:
            logger.info(f"  Auto-Pruned {pruned_count} features from plan.")
        
        # --- Apply SHAP filtering to training data if SHAP JSON now exists ---
        use_shap_filter = opt_conf.get('use_shap_filter', False)
        if use_shap_filter:
            shap_json_path = shap_dir / 'feature_importance.json'
            if shap_json_path.exists():
                logger.info("  Applying SHAP filtering to training features (post-SHAP analysis)...")
                shap_filter_threshold = opt_conf.get('shap_filter_threshold', 0.005)
                
                # Filter training features (full training set)
                X_train_filtered, dropped_train = filter_features_by_shap(
                    X_train,
                    shap_json_path,
                    threshold=shap_filter_threshold
                )
                if dropped_train:
                    logger.info(f"  ✓ Filtered {len(dropped_train)} features from training set")
                    X_train = X_train_filtered
                    
                    # Re-split training data with filtered features
                    logger.info("  Re-splitting training data with filtered features...")
                    X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(
                        X_train, y_train, train_ids, test_size=0.2, random_state=42
                    )
                    logger.info("  ✓ Training and validation features aligned with SHAP filtering")
                    
                    # Re-train model with filtered features
                    logger.info("  Re-training model with SHAP-filtered features...")
                    model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_tr, y_tr), (X_val, y_val)],
                        verbose=False
                    )
                    logger.info("  ✓ Model re-trained with filtered features")
                    
                    # Re-generate validation predictions with filtered model
                    logger.info("  Re-generating validation predictions...")
                    val_preds = model.predict(X_val)
                    val_probs = model.predict_proba(X_val)[:, 1]
        
        # --- Feature Interaction Analysis with H-statistic ---
        if opt_conf.get('analyze_interactions', False):
            if not INTERACTION_ANALYSIS_AVAILABLE:
                logger.warning("  ⚠ Feature interaction analysis not available, skipping...")
            else:
                logger.info("\n[Phase 3b] Feature Interaction Analysis with H-statistic")
                try:
                    # Prepare output directory
                    interaction_output_dir = str(shap_dir / 'interactions')
                    
                    # Get feature importance path
                    feature_importance_path = str(shap_dir / 'feature_importance.csv')
                    
                    # Run interaction analysis
                    interaction_result = analyze_feature_interactions(
                        model=model,
                        X=X_val,
                        feature_importance_path=feature_importance_path,
                        output_dir=interaction_output_dir,
                        top_n_features=opt_conf.get('top_n_features', 10),
                        top_k_pairs=opt_conf.get('top_k_pairs', 5),
                        grid_resolution=opt_conf.get('h_statistic_grid_resolution', 20),
                        sample_size=opt_conf.get('h_statistic_sample_size', 1000),
                        generate_plots=True,
                        test_auc=opt_conf.get('test_interaction_auc', True),
                        model_class=xgb.XGBClassifier,
                        model_params=params,
                        X_train=X_tr,
                        y_train=y_tr,
                        X_val=X_val,
                        y_val=y_val,
                        shap_values=shap_res.get('shap_values'),
                        feature_names=X_train.columns.tolist()
                    )
                    
                    logger.info(f"  ✓ Feature interaction analysis complete")
                    if interaction_result.get('auc_results'):
                        auc_improvement = interaction_result['auc_results'].get('auc_improvement', 0)
                        if auc_improvement > 0:
                            logger.info(f"  ✓ AUC improved by {auc_improvement:.6f} with interaction features")
                        else:
                            logger.info(f"  ⚠ AUC did not improve with interaction features")
                    
                except Exception as e:
                    logger.warning(f"  ⚠ Error in feature interaction analysis: {e}")
                    import traceback
                    logger.debug(f"  Traceback: {traceback.format_exc()}")

    # 5. Prediction (Test)
    logger.info("\n[Phase 4] Test Prediction")
    logger.info("  Loading Test Data...")
    df_test_raw = dp.load_data(str(test_file), clean_text=False)
    
    _, test_non_text = dp.split_text_nontext(
        df_test_raw, prep_conf.get('text_columns', [])
    )
    
    df_test_eng, _ = dp.engineer_features(
        test_non_text,
        config=config,
        is_train=False,
        transformers=transformers,
        hashtag_lib=libs['hashtag_lib'],
        mention_lib=libs['mention_lib']
    )
    
    # Semantic (Test)
    # Note: Test embeddings logic would follow here if configured
    
    X_test, _, _ = dp.prepare_xy(df_test_eng)
    
    # Extract IDs from raw test data (similar to training set)
    if 'challenge_id' in df_test_raw.columns:
        test_ids = df_test_raw['challenge_id']
    elif 'id_str' in df_test_raw.columns:
        test_ids = df_test_raw['id_str']
    elif 'id' in df_test_raw.columns:
        test_ids = df_test_raw['id']
    else:
        test_ids = pd.Series(range(len(df_test_eng))).astype(str)
    
    # Predict
    logger.info("  Predicting...")
    # Align columns
    X_test = X_test[X_train.columns] 
    preds = model.predict(X_test)
    
    save_submission(preds, test_ids, xg_conf['output_file'])
    logger.info("Pipeline Complete.")
    
    # Print CV scores after submission is saved
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    print(f"  Mean CV Accuracy: {mean_cv_score:.4f} ({mean_cv_score*100:.2f}%)")
    print(f"  Std Deviation: {std_cv_score:.4f} ({std_cv_score*100:.2f}%)")
    print(f"  Individual Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean Best Iterations: {np.mean(best_iterations):.1f}")
    print("="*60)

if __name__ == "__main__":
    main()

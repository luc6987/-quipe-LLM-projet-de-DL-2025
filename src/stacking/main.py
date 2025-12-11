"""
Stacking Ensemble for Three Boosting Models
LightGBM + XGBoost + CatBoost Stacking with Rank Averaging

This script implements a 5-Fold Stratified Cross-Validation strategy
for three boosting models: LightGBM, XGBoost, and CatBoost (GPU).
It generates Out-of-Fold (OOF) predictions, computes Pearson correlations,
and uses rank averaging (instead of Meta Model) for final predictions.
Rank averaging is more robust to distribution shift between train and test sets.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.paths import DATA_DIR, get_submissions_dir, get_output_dir
from src.preprocessing import data_processing as dp
from src.utils.config import load_config
from src.xgboost import utils as xgb_utils
from src.lgbm_integration import utils as lgbm_utils
from src.catboost_integration import utils as catboost_utils

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


def generate_oof_predictions(model_name, train_df, test_df, n_folds=5, seed=42):
    """
    Generate OOF predictions for a given boosting model.
    
    Args:
        model_name: 'lightgbm', 'xgboost', or 'catboost'
        train_df: Training DataFrame
        test_df: Test DataFrame
        n_folds: Number of CV folds
        seed: Random seed
        
    Returns:
        oof_logits: OOF predictions (logits) for training set
        test_logits_avg: Average test predictions (logits) across folds
    """
    print(f"\n{'='*60}")
    print(f"Generating OOF predictions for {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config()
    
    # Feature engineering
    train_df_eng, transformers = dp.engineer_features(
        train_df, 
        config=config,
        is_train=True
    )
    test_df_eng, _ = dp.engineer_features(
        test_df, 
        config=config,
        is_train=False, 
        transformers=transformers
    )
    
    # Prepare X, y
    X_train, y_train, _ = dp.prepare_xy(train_df_eng, target_col='label')
    X_test, _, _ = dp.prepare_xy(test_df_eng)
    
    # Align columns
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    X_test = X_test[X_train.columns]
    
    # OOF containers
    oof_logits = np.zeros(len(train_df))
    test_logits_accum = np.zeros(len(test_df))
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Get model-specific parameters
    if model_name == 'xgboost':
        model_config = config.get('xgboost', {})
        params = model_config.get('params', xgb_utils.get_default_params())
        # GPU detection
        gpu_available, _, device_params = xgb_utils.detect_gpu_for_xgboost()
        if gpu_available:
            params.update(device_params)
    elif model_name == 'lightgbm':
        model_config = config.get('lightgbm', {})
        params = model_config.get('params', {})
        # GPU detection
        gpu_available, gpu_params = lgbm_utils.detect_gpu_for_lightgbm()
        if model_config.get('use_gpu', False) and gpu_available:
            params = lgbm_utils.build_lgbm_params(params, use_gpu=True, gpu_params=gpu_params)
        else:
            params = lgbm_utils.build_lgbm_params(params, use_gpu=False)
    elif model_name == 'catboost':
        model_config = config.get('catboost', {})
        params = model_config.get('params', {})
        # GPU detection
        gpu_available, gpu_params = catboost_utils.detect_gpu_for_catboost()
        if model_config.get('use_gpu', False) and gpu_available:
            params = catboost_utils.build_catboost_params(params, use_gpu=True, gpu_params=gpu_params)
        else:
            params = catboost_utils.build_catboost_params(params, use_gpu=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train model
        if model_name == 'xgboost':
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            # Predict logits
            val_logits = model.predict(X_fold_val, output_margin=True)
            test_logits = model.predict(X_test, output_margin=True)
            
        elif model_name == 'lightgbm':
            model = lgbm_utils.create_lgbm_classifier(params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_names=['valid'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            # Predict logits (raw_score=True for logits)
            val_logits = model.predict(X_fold_val, raw_score=True)
            test_logits = model.predict(X_test, raw_score=True)
            
        elif model_name == 'catboost':
            model = catboost_utils.create_catboost_classifier(params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                early_stopping_rounds=50,
                verbose=False
            )
            # Predict logits (prediction_type='RawFormulaVal' for logits)
            val_logits = model.predict(X_fold_val, prediction_type='RawFormulaVal')
            test_logits = model.predict(X_test, prediction_type='RawFormulaVal')
        
        # Store predictions
        oof_logits[val_idx] = val_logits
        test_logits_accum += test_logits
    
    test_logits_avg = test_logits_accum / n_folds
    
    print(f"  ✓ {model_name.upper()} OOF predictions generated")
    print(f"    OOF logits shape: {oof_logits.shape}")
    print(f"    Test logits shape: {test_logits_avg.shape}")
    
    return oof_logits, test_logits_avg


def compute_correlations(oof_predictions_dict, test_predictions_dict=None):
    """
    Compute Pearson correlation matrix between model predictions.
    
    Args:
        oof_predictions_dict: Dict of {model_name: oof_logits}
        test_predictions_dict: Optional dict of {model_name: test_logits}
        
    Returns:
        corr_df: DataFrame with correlation matrix
    """
    print(f"\n{'='*60}")
    print("Computing Pearson Correlations")
    print(f"{'='*60}")
    
    model_names = list(oof_predictions_dict.keys())
    n_models = len(model_names)
    
    # OOF correlations
    corr_matrix_oof = np.zeros((n_models, n_models))
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                corr_matrix_oof[i, j] = 1.0
            else:
                corr, _ = pearsonr(
                    oof_predictions_dict[name1],
                    oof_predictions_dict[name2]
                )
                corr_matrix_oof[i, j] = corr
    
    corr_df_oof = pd.DataFrame(
        corr_matrix_oof,
        index=model_names,
        columns=model_names
    )
    
    print("\nOOF Predictions Correlation Matrix:")
    print(corr_df_oof.round(4))
    
    # Test correlations (if provided)
    if test_predictions_dict:
        corr_matrix_test = np.zeros((n_models, n_models))
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i == j:
                    corr_matrix_test[i, j] = 1.0
                else:
                    corr, _ = pearsonr(
                        test_predictions_dict[name1],
                        test_predictions_dict[name2]
                    )
                    corr_matrix_test[i, j] = corr
        
        corr_df_test = pd.DataFrame(
            corr_matrix_test,
            index=model_names,
            columns=model_names
        )
        
        print("\nTest Predictions Correlation Matrix:")
        print(corr_df_test.round(4))
        
        return corr_df_oof, corr_df_test
    
    return corr_df_oof, None


def run_stacking():
    """
    Main stacking function that can be called from main.py
    """
    print("="*60)
    print("Stacking Ensemble: LightGBM + XGBoost + CatBoost")
    print("="*60)
    
    # Load config
    config = load_config()
    stacking_config = config.get('stacking', {})
    n_folds = stacking_config.get('n_folds', 5)
    seed = 42
    
    # Load data
    train_df = dp.load_data(str(DATA_DIR / 'train.jsonl'), clean_text=True)
    test_df = dp.load_data(str(DATA_DIR / 'kaggle_test.jsonl'), clean_text=True)
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Generate OOF predictions for all three models
    models = ['lightgbm', 'xgboost', 'catboost']
    oof_predictions = {}
    test_predictions = {}
    
    for model_name in models:
        oof_logits, test_logits = generate_oof_predictions(
            model_name, train_df, test_df, n_folds=n_folds, seed=seed
        )
        oof_predictions[model_name] = oof_logits
        test_predictions[model_name] = test_logits
    
    # Compute correlations
    corr_df_oof, corr_df_test = compute_correlations(
        oof_predictions, test_predictions
    )
    
    # Save correlations
    output_dir = get_output_dir('stacking')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corr_oof_path = output_dir / 'correlations_oof.csv'
    corr_df_oof.to_csv(corr_oof_path)
    print(f"\n✓ OOF correlations saved to: {corr_oof_path}")
    
    if corr_df_test is not None:
        corr_test_path = output_dir / 'correlations_test.csv'
        corr_df_test.to_csv(corr_test_path)
        print(f"✓ Test correlations saved to: {corr_test_path}")
    
    # Save OOF predictions for reuse
    oof_predictions_path = output_dir / 'oof_predictions.npz'
    np.savez(oof_predictions_path, **oof_predictions)
    print(f"✓ OOF predictions saved to: {oof_predictions_path}")
    
    test_predictions_path = output_dir / 'test_predictions.npz'
    np.savez(test_predictions_path, **test_predictions)
    print(f"✓ Test predictions saved to: {test_predictions_path}")
    
    # Prepare meta-features (for optional Meta Model training/diagnosis)
    print(f"\n{'='*60}")
    print("Ensemble Strategy: Rank Averaging (No Meta Model)")
    print(f"{'='*60}")
    print(f"\nNote: Meta Model training is skipped. Using rank averaging instead.")
    
    # Get labels
    y_train_labels = train_df['label'].values
    
    # Check model correlations
    max_correlation = max([
        corr_df_oof.loc['lightgbm', 'xgboost'],
        corr_df_oof.loc['lightgbm', 'catboost'],
        corr_df_oof.loc['xgboost', 'catboost']
    ])
    
    print(f"\nMaximum correlation between models: {max_correlation:.4f}")
    if max_correlation > 0.95:
        print("  ⚠ Warning: Very high correlation detected. Rank averaging is a good choice.")
    
    # Optional: Train Meta Model for comparison (but won't be used for final prediction)
    # This can help diagnose if Meta Model would work better
    train_meta_model_for_diagnosis = False  # Set to True if you want to compare
    
    if train_meta_model_for_diagnosis:
        print(f"\n{'='*60}")
        print("Training Meta-Model (for diagnosis only, not used)")
        print(f"{'='*60}")
        
        X_meta = np.column_stack([
            oof_predictions['lightgbm'],
            oof_predictions['xgboost'],
            oof_predictions['catboost']
        ])
        
        X_meta_train, X_meta_val, y_train_split, y_val_split = train_test_split(
            X_meta, y_train_labels,
            test_size=0.2,
            random_state=seed,
            stratify=y_train_labels
        )
        
        scaler = StandardScaler()
        X_meta_train_scaled = scaler.fit_transform(X_meta_train)
        X_meta_val_scaled = scaler.transform(X_meta_val)
        
        meta_model = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            C=0.1,
            penalty='l2'
        )
        meta_model.fit(X_meta_train_scaled, y_train_split)
        
        val_preds = meta_model.predict_proba(X_meta_val_scaled)[:, 1]
        best_acc = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 81):
            acc = accuracy_score(y_val_split, val_preds > thresh)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        val_acc = accuracy_score(y_val_split, val_preds > best_thresh)
        val_auc = roc_auc_score(y_val_split, val_preds)
        
        print(f"Meta-Model (diagnosis): Accuracy={val_acc:.4f}, AUC={val_auc:.4f}, Threshold={best_thresh:.4f}")
        
        # Also get test predictions for comparison
        X_meta_test = np.column_stack([
            test_predictions['lightgbm'],
            test_predictions['xgboost'],
            test_predictions['catboost']
        ])
        X_meta_test_scaled = scaler.transform(X_meta_test)
        test_preds = meta_model.predict_proba(X_meta_test_scaled)[:, 1]
    else:
        best_thresh = 0.5  # Default, not used
        val_acc = 0.0
        val_auc = 0.0
        test_preds = None
    
    # Check base model performance (OOF AUC for each model)
    print(f"\n{'='*60}")
    print("Base Model Performance (OOF)")
    print(f"{'='*60}")
    train_positive_ratio = np.mean(y_train_labels)
    print(f"Training set positive ratio: {train_positive_ratio:.4f}")
    
    base_model_performance = {}
    for model_name in models:
        # Convert logits to probabilities
        oof_probs = 1 / (1 + np.exp(-oof_predictions[model_name]))
        oof_auc = roc_auc_score(y_train_labels, oof_probs)
        base_model_performance[model_name] = {
            'oof_auc': float(oof_auc),
            'oof_mean_prob': float(np.mean(oof_probs)),
            'oof_median_prob': float(np.median(oof_probs))
        }
        print(f"{model_name}: OOF AUC = {oof_auc:.4f}, Mean prob = {np.mean(oof_probs):.4f}")
    
    # Test set predictions will be done using rank averaging (see below)
    
    # ====================================================================
    # Base Model Test Set Prediction Diagnosis
    # ====================================================================
    print(f"\n{'='*60}")
    print("Base Model Test Set Prediction Analysis")
    print(f"{'='*60}")
    
    base_test_probs = {}
    for model_name in models:
        # Convert test logits to probabilities
        test_logits = test_predictions[model_name]
        test_probs = 1 / (1 + np.exp(-test_logits))
        base_test_probs[model_name] = test_probs
        
        print(f"\n{model_name.upper()} Test Set Predictions:")
        print(f"  Logits: mean={np.mean(test_logits):.4f}, median={np.median(test_logits):.4f}, std={np.std(test_logits):.4f}")
        print(f"  Probabilities: mean={np.mean(test_probs):.4f}, median={np.median(test_probs):.4f}, std={np.std(test_probs):.4f}")
        print(f"  Percentiles: 25%={np.percentile(test_probs, 25):.4f}, "
              f"50%={np.percentile(test_probs, 50):.4f}, "
              f"75%={np.percentile(test_probs, 75):.4f}")
        
        # Compare with OOF
        oof_logits = oof_predictions[model_name]
        oof_probs = 1 / (1 + np.exp(-oof_logits))
        logit_diff = np.mean(test_logits) - np.mean(oof_logits)
        prob_diff = np.mean(test_probs) - np.mean(oof_probs)
        print(f"  vs OOF: logit_diff={logit_diff:.4f}, prob_diff={prob_diff:.4f}")
        
        if abs(logit_diff) > 1.0 or abs(prob_diff) > 0.2:
            print(f"  ⚠️  WARNING: Large difference between test and OOF predictions!")
    
    # ====================================================================
    # Distribution Shift Analysis (using base model probabilities)
    # ====================================================================
    # Analyze distribution shift using OOF vs test probabilities from base models
    print(f"\n{'='*60}")
    print("Distribution Shift Analysis (Base Models)")
    print(f"{'='*60}")
    
    # Compare OOF probabilities vs test probabilities for each model
    oof_probs_avg = np.mean([1/(1+np.exp(-oof_predictions[m])) for m in models], axis=0)
    test_probs_avg = np.mean([base_test_probs[m] for m in models], axis=0)
    
    mean_diff = np.mean(test_probs_avg) - np.mean(oof_probs_avg)
    median_diff = np.median(test_probs_avg) - np.median(oof_probs_avg)
    
    print(f"\nOOF Probabilities (Average across models):")
    print(f"  Mean: {np.mean(oof_probs_avg):.4f}")
    print(f"  Median: {np.median(oof_probs_avg):.4f}")
    print(f"  Std: {np.std(oof_probs_avg):.4f}")
    
    print(f"\nTest Probabilities (Average across models):")
    print(f"  Mean: {np.mean(test_probs_avg):.4f}")
    print(f"  Median: {np.median(test_probs_avg):.4f}")
    print(f"  Std: {np.std(test_probs_avg):.4f}")
    
    print(f"\n⚠️  Distribution Shift:")
    print(f"  Mean difference (test - OOF): {mean_diff:.4f}")
    print(f"  Median difference (test - OOF): {median_diff:.4f}")
    
    if abs(mean_diff) > 0.1:
        print(f"  ⚠️  WARNING: Large distribution shift detected!")
        print(f"      This confirms why rank averaging is the right choice.")
    elif abs(mean_diff) > 0.05:
        print(f"  ⚠️  Moderate distribution shift detected.")
    else:
        print(f"  ✓ Distribution looks similar.")
    
    # ====================================================================
    # Rank Averaging Strategy (No Meta Model)
    # ====================================================================
    print(f"\n{'='*60}")
    print("Rank Averaging Strategy (No Meta Model)")
    print(f"{'='*60}")
    
    print(f"\nUsing Rank Averaging instead of Meta Model:")
    print(f"  Rank averaging is more robust to distribution shift")
    print(f"  and doesn't require probability calibration.")
    
    # Convert test logits to probabilities for ranking
    test_probs_dict = {}
    for model_name in models:
        test_logits = test_predictions[model_name]
        test_probs = 1 / (1 + np.exp(-test_logits))
        test_probs_dict[model_name] = test_probs
    
    # Calculate normalized ranks for each model
    print(f"\nCalculating normalized ranks...")
    test_ranks_dict = {}
    for model_name in models:
        probs = test_probs_dict[model_name]
        # Normalized rank: rank / len (gives values in [0, 1])
        ranks = rankdata(probs) / len(probs)
        test_ranks_dict[model_name] = ranks
        print(f"  {model_name}: rank mean={np.mean(ranks):.4f}, std={np.std(ranks):.4f}")
    
    # Calculate weights based on OOF AUC (better models get higher weights)
    print(f"\nCalculating model weights based on OOF AUC:")
    oof_aucs = {m: base_model_performance[m]['oof_auc'] for m in models}
    total_auc = sum(oof_aucs.values())
    model_weights = {m: oof_aucs[m] / total_auc for m in models}
    
    for model_name in models:
        print(f"  {model_name}: weight={model_weights[model_name]:.4f} (OOF AUC={oof_aucs[model_name]:.4f})")
    
    # Weighted rank averaging
    final_ranks = np.zeros(len(test_probs_dict[models[0]]))
    for model_name in models:
        final_ranks += test_ranks_dict[model_name] * model_weights[model_name]
    
    print(f"\nFinal rank statistics:")
    print(f"  Mean: {np.mean(final_ranks):.4f}")
    print(f"  Median: {np.median(final_ranks):.4f}")
    print(f"  Std: {np.std(final_ranks):.4f}")
    print(f"  Min: {np.min(final_ranks):.4f}, Max: {np.max(final_ranks):.4f}")
    
    # Convert ranks back to probabilities (ranks are already in [0,1])
    test_preds_final = final_ranks
    
    # Use percentile threshold to match training set positive ratio
    final_threshold = np.percentile(test_preds_final, (1 - train_positive_ratio) * 100)
    threshold_strategy = "rank_averaging_percentile"
    
    print(f"\nUsing percentile-based threshold:")
    print(f"  Target positive ratio: {train_positive_ratio:.4f}")
    print(f"  Percentile threshold: {final_threshold:.4f}")
    print(f"  (This ensures ~{train_positive_ratio*100:.1f}% of test samples are predicted as positive)")
    
    # Apply threshold
    binary_preds = (test_preds_final >= final_threshold).astype(int)
    
    # Report final prediction distribution
    positive_count = binary_preds.sum()
    positive_ratio = positive_count / len(binary_preds)
    print(f"\nFinal Prediction Distribution:")
    print(f"  Positive predictions: {positive_count}/{len(binary_preds)} ({positive_ratio*100:.2f}%)")
    print(f"  Negative predictions: {(len(binary_preds)-positive_count)}/{len(binary_preds)} ({(1-positive_ratio)*100:.2f}%)")
    
    # Test different thresholds for analysis
    print(f"\nThreshold Sensitivity Analysis (test set - rank averaging):")
    threshold_options = [0.3, 0.4, 0.5, 0.6, final_threshold]
    threshold_options = sorted(set(threshold_options))  # Remove duplicates and sort
    for thresh in threshold_options:
        ratio = (test_preds_final > thresh).sum() / len(test_preds_final)
        marker = " <- selected" if abs(thresh - final_threshold) < 0.01 else ""
        print(f"  Threshold {thresh:.4f}: {ratio*100:5.2f}% predicted as positive{marker}")
    
    # Show alternative strategies for comparison
    print(f"\nAlternative Strategies (for comparison):")
    
    # Meta Model option (only if it was trained)
    if train_meta_model_for_diagnosis and 'test_preds' in locals() and test_preds is not None:
        print(f"\n  Meta Model predictions:")
        for thresh in [0.3, 0.4, 0.5]:
            ratio = (test_preds > thresh).sum() / len(test_preds)
            print(f"    Threshold {thresh:.2f}: {ratio*100:5.2f}% predicted as positive")
    else:
        print(f"\n  Meta Model predictions: Not available (Meta Model was not trained)")
    
    # Simple probability average option
    base_avg_probs = np.mean([base_test_probs[m] for m in models], axis=0)
    print(f"\n  Simple probability average:")
    for thresh in [0.3, 0.4, 0.5]:
        ratio = (base_avg_probs > thresh).sum() / len(base_avg_probs)
        print(f"    Threshold {thresh:.2f}: {ratio*100:5.2f}% predicted as positive")
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_df['challenge_id'].values,
        'Prediction': binary_preds
    })
    
    output_file = get_submissions_dir() / 'submission_stacking_boosting.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n✓ Submission saved to: {output_file}")
    print(f"  Predictions distribution: {submission['Prediction'].value_counts().sort_index().to_dict()}")
    
    # Save meta-model info
    meta_info = {
        'final_threshold': float(final_threshold),
        'threshold_strategy': threshold_strategy,
        'train_positive_ratio': float(train_positive_ratio),
        'test_positive_ratio': float(positive_ratio),
        'max_correlation': float(max_correlation),
        'distribution_shift': {
            'mean_diff': float(mean_diff),
            'median_diff': float(median_diff)
        },
        'model_weights': model_weights,
        'strategy': 'rank_averaging',
        'rank_statistics': {
            'mean': float(np.mean(final_ranks)),
            'median': float(np.median(final_ranks)),
            'std': float(np.std(final_ranks)),
            'min': float(np.min(final_ranks)),
            'max': float(np.max(final_ranks))
        },
        'base_model_performance': base_model_performance,
        'n_folds': n_folds,
        'models': models
    }
    
    # Add optional Meta Model info if it was trained
    if train_meta_model_for_diagnosis:
        meta_info['meta_model_diagnosis'] = {
            'validation_accuracy': float(val_acc),
            'validation_auc': float(val_auc),
            'best_threshold': float(best_thresh)
        }
    
    import json
    meta_info_path = output_dir / 'meta_model_info.json'
    with open(meta_info_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"✓ Meta-model info saved to: {meta_info_path}")
    
    print(f"\n{'='*60}")
    print("Stacking Complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - Correlations: {corr_oof_path}")
    print(f"  - OOF predictions: {oof_predictions_path}")
    print(f"  - Test predictions: {test_predictions_path}")
    print(f"  - Submission: {output_file}")
    print(f"  - Meta-model info: {meta_info_path}")


def main():
    """Entry point when run as script"""
    run_stacking()


if __name__ == "__main__":
    main()


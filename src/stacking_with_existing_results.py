# 创建文件: src/stacking_with_existing_results.py

"""
Quick Stacking using Existing Embeddings and XGBoost Predictions
使用现有结果进行快速Stacking测试
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.paths import get_submissions_dir, get_embeddings_dir, DATA_DIR
from src.utils import data_processing as dp
import xgboost as xgb
from src.xgboost import xgboost_baseline

def load_embeddings(embedding_path):
    """Load embeddings from pickle file."""
    with open(embedding_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data.get('labels', None)

def generate_xgboost_oof_predictions(train_df, test_df, n_folds=5, seed=42):
    """
    Generate XGBoost OOF predictions using cross-validation.
    Returns: oof_train_logits, test_logits_avg
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("Generating XGBoost OOF predictions...")
    
    # Feature engineering
    train_df_eng, transformers = dp.engineer_features(train_df, is_train=True)
    test_df_eng, _ = dp.engineer_features(test_df, is_train=False, transformers=transformers)
    
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
    params = xgboost_baseline.get_default_params()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train XGBoost
        model = xgb.XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train, 
                 eval_set=[(X_fold_val, y_fold_val)],
                 verbose=False)
        
        # Predict logits
        val_logits = model.predict(X_fold_val, output_margin=True)
        test_logits = model.predict(X_test, output_margin=True)
        
        oof_logits[val_idx] = val_logits
        test_logits_accum += test_logits
    
    test_logits_avg = test_logits_accum / n_folds
    return oof_logits, test_logits_avg

def main():
    print("="*60)
    print("Quick Stacking with Existing Embeddings")
    print("="*60)
    
    # 1. Load existing embeddings
    embedding_dir = get_embeddings_dir()
    print("\n1. Loading existing embeddings...")
    
    train_emb_path = str(embedding_dir / "train_embeddings.pkl")
    val_emb_path = str(embedding_dir / "val_embeddings.pkl")
    test_emb_path = str(embedding_dir / "test_embeddings.pkl")
    
    if not all(os.path.exists(p) for p in [train_emb_path, val_emb_path, test_emb_path]):
        print("Error: Embedding files not found!")
        print(f"  Expected: {train_emb_path}")
        print(f"  Expected: {val_emb_path}")
        print(f"  Expected: {test_emb_path}")
        return
    
    train_emb, train_labels = load_embeddings(train_emb_path)
    val_emb, val_labels = load_embeddings(val_emb_path)
    test_emb, _ = load_embeddings(test_emb_path)
    
    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Val embeddings: {val_emb.shape}")
    print(f"  Test embeddings: {test_emb.shape}")
    
    # Combine train + val embeddings (to match full training set)
    full_train_emb = np.vstack([train_emb, val_emb])
    full_train_labels = np.concatenate([train_labels, val_labels])
    
    print(f"  Combined train embeddings: {full_train_emb.shape}")
    
    # 2. Generate XGBoost OOF predictions
    print("\n2. Generating XGBoost OOF predictions...")
    train_df = dp.load_data(str(DATA_DIR / 'train.jsonl'), clean_text=True)
    test_df = dp.load_data(str(DATA_DIR / 'kaggle_test.jsonl'), clean_text=True)
    
    xgb_oof_logits, xgb_test_logits = generate_xgboost_oof_predictions(train_df, test_df)
    
    # 3. Prepare meta-features
    print("\n3. Preparing meta-features...")
    
    # Reduce embedding dimensions
    text_components = 128
    reducer = TruncatedSVD(n_components=min(text_components, full_train_emb.shape[1]-1), random_state=42)
    reduced_train_emb = reducer.fit_transform(full_train_emb)
    reduced_test_emb = reducer.transform(test_emb)
    
    print(f"  Reduced train embeddings: {reduced_train_emb.shape}")
    print(f"  Reduced test embeddings: {reduced_test_emb.shape}")
    
    # Combine XGBoost logits + embeddings
    X_meta_train = np.column_stack([xgb_oof_logits, reduced_train_emb])
    X_meta_test = np.column_stack([xgb_test_logits, reduced_test_emb])
    
    # Scale
    scaler = StandardScaler()
    X_meta_train_scaled = scaler.fit_transform(X_meta_train)
    X_meta_test_scaled = scaler.transform(X_meta_test)
    
    # 4. Train meta-model
    print("\n4. Training meta-model (Logistic Regression)...")
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(X_meta_train_scaled, full_train_labels)
    
    # Evaluate on training set
    train_preds = meta_model.predict_proba(X_meta_train_scaled)[:, 1]
    
    # Find optimal threshold
    best_acc = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.1, 0.9, 81):
        acc = accuracy_score(full_train_labels, train_preds > thresh)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    acc = accuracy_score(full_train_labels, train_preds > best_thresh)
    auc = roc_auc_score(full_train_labels, train_preds)
    
    print(f"  Training Accuracy: {acc:.4f} (Threshold: {best_thresh:.4f})")
    print(f"  Training AUC: {auc:.4f}")
    
    # 5. Predict test set
    print("\n5. Generating test predictions...")
    test_preds = meta_model.predict_proba(X_meta_test_scaled)[:, 1]
    binary_preds = (test_preds >= best_thresh).astype(int)
    
    # 6. Create submission
    submission = pd.DataFrame({
        'ID': test_df['challenge_id'].values,
        'Prediction': binary_preds
    })
    
    output_file = str(get_submissions_dir() / 'submission_stacking_quick.csv')
    submission.to_csv(output_file, index=False)
    print(f"\n✓ Submission saved to: {output_file}")
    print(f"  Predictions distribution: {submission['Prediction'].value_counts().sort_index().to_dict()}")

if __name__ == "__main__":
    main()
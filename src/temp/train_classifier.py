"""
Classifier Training Module for Distribution Shift Detection
分布偏移检测的分类器训练模块

Trains an XGBoost classifier to distinguish between training and test sets.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # Placeholder for type hints
    logging.warning("XGBoost not available. Please install xgboost.")

logger = logging.getLogger(__name__)


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    xgb_params: Optional[Dict] = None,
    save_model: bool = False,
    model_path: Optional[Path] = None
) -> Tuple[Any, Dict]:
    """
    Train an XGBoost classifier to distinguish between train and test sets.
    
    Args:
        X: Feature DataFrame
        y: Labels (0 for train, 1 for test)
        n_splits: Number of folds for cross-validation
        random_state: Random seed
        xgb_params: XGBoost parameters. If None, uses default.
        save_model: Whether to save the trained model
        model_path: Path to save the model. If None, uses default.
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Please install xgboost.")
    
    logger.info("="*60)
    logger.info("Training XGBoost Classifier for Distribution Shift Detection")
    logger.info("="*60)
    
    # Default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    logger.info(f"XGBoost parameters: {xgb_params}")
    
    # Initialize classifier
    clf = xgb.XGBClassifier(**xgb_params)
    
    # Cross-validation
    logger.info(f"\nPerforming {n_splits}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Cross-validation scores
    cv_scores_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores_acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    logger.info(f"Cross-validation AUC scores: {cv_scores_auc}")
    logger.info(f"  Mean AUC: {cv_scores_auc.mean():.4f} (+/- {cv_scores_auc.std() * 2:.4f})")
    logger.info(f"Cross-validation Accuracy scores: {cv_scores_acc}")
    logger.info(f"  Mean Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    
    # Train on full dataset
    logger.info("\nTraining on full dataset...")
    clf.fit(X, y)
    
    # Predictions on full dataset
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    logger.info("\nFull Dataset Performance:")
    logger.info(f"  AUC-ROC: {auc:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives (Train predicted as Train): {cm[0, 0]}")
    logger.info(f"  False Positives (Train predicted as Test): {cm[0, 1]}")
    logger.info(f"  False Negatives (Test predicted as Train): {cm[1, 0]}")
    logger.info(f"  True Positives (Test predicted as Test): {cm[1, 1]}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y, y_pred, target_names=['Train', 'Test']))
    
    # Interpretation
    logger.info("\n" + "="*60)
    if auc > 0.9:
        logger.warning("⚠️  HIGH DISTRIBUTION SHIFT DETECTED!")
        logger.warning(f"   AUC = {auc:.4f} indicates strong ability to distinguish train vs test.")
        logger.warning("   This suggests significant distribution shift in features.")
    elif auc > 0.7:
        logger.warning("⚠️  MODERATE DISTRIBUTION SHIFT DETECTED!")
        logger.warning(f"   AUC = {auc:.4f} indicates some ability to distinguish train vs test.")
        logger.warning("   This suggests moderate distribution shift in features.")
    else:
        logger.info("✓ LOW DISTRIBUTION SHIFT")
        logger.info(f"   AUC = {auc:.4f} indicates limited ability to distinguish train vs test.")
        logger.info("   This suggests minimal distribution shift in features.")
    logger.info("="*60)
    
    # Save model if requested
    if save_model:
        if model_path is None:
            from src.utils.paths import get_output_dir
            output_dir = get_output_dir('temp')
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / 'distribution_shift_classifier.pkl'
        
        logger.info(f"\nSaving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        logger.info("Model saved.")
    
    # Compile metrics
    metrics = {
        'cv_auc_mean': cv_scores_auc.mean(),
        'cv_auc_std': cv_scores_auc.std(),
        'cv_accuracy_mean': cv_scores_acc.mean(),
        'cv_accuracy_std': cv_scores_acc.std(),
        'full_auc': auc,
        'full_accuracy': accuracy,
        'full_precision': precision,
        'full_recall': recall,
        'full_f1': f1,
        'confusion_matrix': cm.tolist(),
        'n_splits': n_splits,
        'n_features': len(X.columns),
        'n_samples': len(X)
    }
    
    return clf, metrics


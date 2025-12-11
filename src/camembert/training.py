"""
CamemBERT Training Pipeline Module
CamemBERT 

Provides training pipeline integration with XGBoost and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Training functions will be limited.")


def prepare_features_with_camembert(
    df: pd.DataFrame,
    transformers: Optional[Dict] = None,
    is_train: bool = True,
    **camembert_kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare features using CamemBERT embeddings.
    CamemBERT
    
    This is a wrapper that integrates CamemBERT processing into the feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        transformers: Dictionary of transformers
        is_train: Whether this is training data
        **camembert_kwargs: Additional arguments for process_camembert_embeddings
        
    Returns:
        Tuple of (feature DataFrame, transformers dict)
    """
    from .embedding import process_camembert_embeddings
    
    logger.info("Preparing features with CamemBERT embeddings...")
    
    df_processed, transformers = process_camembert_embeddings(
        df,
        is_train=is_train,
        transformers=transformers,
        **camembert_kwargs
    )
    
    return df_processed, transformers


def train_with_camembert_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    groups: Optional[np.ndarray] = None,
    xgb_params: Optional[Dict] = None,
    n_folds: int = 5,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Train XGBoost model with CamemBERT features.
    CamemBERTXGBoost
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        groups: Optional group labels for GroupKFold
        xgb_params: XGBoost parameters
        n_folds: Number of CV folds
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary with training results and model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required for training")
    
    logger.info("Training XGBoost with CamemBERT features...")
    
    # Default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error', 'auc'],
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'min_child_weight': 5,
            'n_estimators': 400,
            'gamma': 0.2,
            'reg_alpha': 0.15,
            'reg_lambda': 1.5,
            'random_state': 42,
        }
    
    # Prepare training data
    train_data = xgb.DMatrix(X_train, label=y_train)
    
    # Prepare validation data if provided
    val_data = None
    if X_val is not None and y_val is not None:
        val_data = xgb.DMatrix(X_val, label=y_val)
        evals = [(train_data, 'train'), (val_data, 'val')]
    else:
        evals = [(train_data, 'train')]
    
    # Train model
    model = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=xgb_params.get('n_estimators', 400),
        evals=evals,
        early_stopping_rounds=50 if val_data else None,
        verbose_eval=10
    )
    
    # Make predictions
    train_pred = model.predict(train_data)
    train_pred_binary = (train_pred > 0.5).astype(int)
    train_auc = roc_auc_score(y_train, train_pred)
    train_acc = accuracy_score(y_train, train_pred_binary)
    
    results = {
        'model': model,
        'train_auc': train_auc,
        'train_acc': train_acc,
        'train_predictions': train_pred,
    }
    
    if val_data is not None:
        val_pred = model.predict(val_data)
        val_pred_binary = (val_pred > 0.5).astype(int)
        val_auc = roc_auc_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred_binary)
        
        results['val_auc'] = val_auc
        results['val_acc'] = val_acc
        results['val_predictions'] = val_pred
        
        logger.info(f"Validation - AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")
    
    logger.info(f"Training - AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    
    # Generate visualizations if output directory provided
    if output_dir:
        try:
            from .visualization import save_training_report
        except ImportError:
            logger.warning("visualization module not available. Skipping training report generation.")
            save_training_report = None
        
        if save_training_report:
            train_scores = {
                'AUC': [train_auc],
                'Accuracy': [train_acc]
            }
            val_scores = None
            if 'val_auc' in results:
                val_scores = {
                    'AUC': [results['val_auc']],
                    'Accuracy': [results['val_acc']]
                }
            
            save_training_report(
                output_dir,
                train_scores=train_scores,
                val_scores=val_scores,
            metadata={
                'model_type': 'XGBoost with CamemBERT',
                'n_features': X_train.shape[1],
                'n_samples': len(X_train),
            }
        )
    
    return results


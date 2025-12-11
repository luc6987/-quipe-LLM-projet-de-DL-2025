"""
Data Loading Module for Distribution Shift Detection
分布偏移检测的数据加载模块

Loads training and test features, creates labels (train=0, test=1), and merges them.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.utils import shuffle

from src.utils.paths import get_preprocessing_dir

logger = logging.getLogger(__name__)


def load_train_test_features(
    train_features_path: Optional[Path] = None,
    test_features_path: Optional[Path] = None,
    balance_samples: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training and test features, create labels, and merge them.
    
    Args:
        train_features_path: Path to train_features.pkl. If None, uses default path.
        test_features_path: Path to test_features.pkl. If None, uses default path.
        balance_samples: If True, balance the number of samples from train and test sets.
        random_state: Random seed for shuffling and sampling.
        
    Returns:
        Tuple of (X_combined, y_combined) where:
        - X_combined: Combined feature DataFrame
        - y_combined: Labels (0 for train, 1 for test)
    """
    # Use default paths if not provided
    if train_features_path is None:
        preprocessing_dir = get_preprocessing_dir()
        train_features_path = preprocessing_dir / "train_features.pkl"
    
    if test_features_path is None:
        preprocessing_dir = get_preprocessing_dir()
        test_features_path = preprocessing_dir / "test_features.pkl"
    
    # Load features
    logger.info(f"Loading training features from {train_features_path}...")
    if not train_features_path.exists():
        raise FileNotFoundError(f"Training features not found at {train_features_path}")
    X_train = pd.read_pickle(train_features_path)
    logger.info(f"  Loaded {len(X_train)} training samples with {len(X_train.columns)} features")
    
    logger.info(f"Loading test features from {test_features_path}...")
    if not test_features_path.exists():
        raise FileNotFoundError(f"Test features not found at {test_features_path}")
    X_test = pd.read_pickle(test_features_path)
    logger.info(f"  Loaded {len(X_test)} test samples with {len(X_test.columns)} features")
    
    # Align features (use intersection of columns)
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    common_cols = sorted(list(train_cols & test_cols))
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols
    
    if train_only_cols:
        logger.warning(f"  {len(train_only_cols)} features only in training set (will be dropped)")
    if test_only_cols:
        logger.warning(f"  {len(test_only_cols)} features only in test set (will be dropped)")
    
    logger.info(f"  Using {len(common_cols)} common features")
    
    # Select common columns
    X_train_aligned = X_train[common_cols].copy()
    X_test_aligned = X_test[common_cols].copy()
    
    # Create labels: 0 for train, 1 for test
    y_train = pd.Series(0, index=X_train_aligned.index, name='source')
    y_test = pd.Series(1, index=X_test_aligned.index, name='source')
    
    # Balance samples if requested
    if balance_samples:
        min_samples = min(len(X_train_aligned), len(X_test_aligned))
        logger.info(f"Balancing samples: selecting {min_samples} samples from each set")
        
        # Sample from the larger set
        if len(X_train_aligned) > min_samples:
            X_train_aligned = X_train_aligned.sample(n=min_samples, random_state=random_state)
            y_train = y_train.loc[X_train_aligned.index]
        
        if len(X_test_aligned) > min_samples:
            X_test_aligned = X_test_aligned.sample(n=min_samples, random_state=random_state)
            y_test = y_test.loc[X_test_aligned.index]
    
    # Combine data
    logger.info("Combining training and test data...")
    X_combined = pd.concat([X_train_aligned, X_test_aligned], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    
    # Shuffle
    logger.info("Shuffling combined data...")
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=random_state)
    
    logger.info(f"Final combined dataset: {len(X_combined)} samples, {len(X_combined.columns)} features")
    logger.info(f"  Train samples (label=0): {(y_combined == 0).sum()}")
    logger.info(f"  Test samples (label=1): {(y_combined == 1).sum()}")
    
    return X_combined, y_combined


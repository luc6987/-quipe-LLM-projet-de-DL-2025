"""
验证策略模块：实现GroupKFold交叉验证 / Validation strategy module: Implements GroupKFold cross-validation
关键：防止数据泄露，确保同一用户的所有推文在同一fold中 / Key: Prevent data leakage, ensure all tweets from same user stay in same fold
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')


def create_group_kfold(n_splits=5, shuffle=False, random_state=42):
    """
    创建GroupKFold交叉验证对象 / Create GroupKFold cross-validation object
    
    Args:
        n_splits: 折数（默认5） / Number of splits (default 5)
        shuffle: 是否打乱组（默认False） / Whether to shuffle groups (default False)
        random_state: 随机种子 / Random seed
        
    Returns:
        GroupKFold对象 / GroupKFold object
    """
    return GroupKFold(n_splits=n_splits, shuffle=shuffle)


def group_kfold_cv(X, y, groups, n_splits=5, model=None, model_params=None, 
                   eval_metric='accuracy', early_stopping_rounds=None, 
                   verbose=True):
    """
    执行GroupKFold交叉验证 / Perform GroupKFold cross-validation
    
    这是防止数据泄露的关键：确保同一用户的所有推文在同一fold中 / This is key to preventing data leakage: ensure all tweets from same user stay in same fold
    
    Args:
        X: 特征DataFrame或numpy数组 / Feature DataFrame or numpy array
        y: 标签Series或numpy数组 / Label Series or numpy array
        groups: 用户ID数组（用于分组） / User ID array (for grouping)
        n_splits: 折数 / Number of splits
        model: 模型对象（如果为None，需要提供model_params） / Model object (if None, need to provide model_params)
        model_params: 模型参数字典 / Model parameter dictionary
        eval_metric: 评估指标（'accuracy'或'logloss'） / Evaluation metric ('accuracy' or 'logloss')
        early_stopping_rounds: 早停轮数（如果模型支持） / Early stopping rounds (if model supports)
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        dict: 包含CV结果的字典 / Dictionary containing CV results
            - 'scores': 各折分数列表 / List of fold scores
            - 'mean_score': 平均分数 / Mean score
            - 'std_score': 标准差 / Standard deviation
            - 'best_iterations': 各折的最佳迭代次数（如果支持） / Best iteration for each fold (if supported)
    """
    if verbose:
        print("\n" + "="*60)
        print(f"GroupKFold Cross-Validation (n_splits={n_splits})")
        print("="*60)
        print(f"  Total samples: {len(X)}")
        print(f"  Unique users: {len(np.unique(groups))}")
        print(f"  Average tweets per user: {len(X) / len(np.unique(groups)):.2f}")
    
    gkf = create_group_kfold(n_splits=n_splits)
    
    scores = []
    best_iterations = []
    
    # 转换为numpy数组（如果还不是）
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
    
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = y
    
    if isinstance(groups, pd.Series):
        groups_array = groups.values
    else:
        groups_array = groups
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_array, y_array, groups=groups_array), 1):
        if verbose:
            print(f"\nFold {fold}/{n_splits}:")
            print(f"  Training set size: {len(train_idx)}")
            print(f"  Validation set size: {len(val_idx)}")
            print(f"  Training set users: {len(np.unique(groups_array[train_idx]))}")
            print(f"  Validation set users: {len(np.unique(groups_array[val_idx]))}")
        
        X_train_fold = X_array[train_idx]
        X_val_fold = X_array[val_idx]
        y_train_fold = y_array[train_idx]
        y_val_fold = y_array[val_idx]
        
        # 创建模型 / Create model
        if model is None:
            if model_params is None:
                raise ValueError("Must provide model or model_params")
            # 假设使用XGBoost（需要根据实际情况调整） / Assume using XGBoost (adjust according to actual situation)
            import xgboost as xgb
            model_fold = xgb.XGBClassifier(**model_params)
        else:
            # 克隆模型 / Clone model
            from sklearn.base import clone
            model_fold = clone(model)
            if model_params:
                model_fold.set_params(**model_params)
        
        # 训练模型 / Train model
        fit_params = {}
        if early_stopping_rounds is not None:
            fit_params['eval_set'] = [(X_val_fold, y_val_fold)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = False
        
        model_fold.fit(X_train_fold, y_train_fold, **fit_params)
        
        # 预测 / Predict
        y_pred = model_fold.predict(X_val_fold)
        
        # 计算分数 / Calculate score
        if eval_metric == 'accuracy':
            score = accuracy_score(y_val_fold, y_pred)
        elif eval_metric == 'logloss':
            y_proba = model_fold.predict_proba(X_val_fold)[:, 1]
            score = -log_loss(y_val_fold, y_proba)  # 负值，因为log_loss越小越好 / Negative value, because smaller log_loss is better
        else:
            raise ValueError(f"Unsupported evaluation metric: {eval_metric}")
        
        scores.append(score)
        
        # 获取最佳迭代次数（如果支持） / Get best iteration (if supported)
        if hasattr(model_fold, 'best_iteration'):
            best_iterations.append(model_fold.best_iteration)
        
        if verbose:
            print(f"  {eval_metric}: {score:.5f}")
    
    results = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'best_iterations': best_iterations if best_iterations else None
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results:")
        print(f"  Mean {eval_metric}: {results['mean_score']:.5f} ± {results['std_score']:.5f}")
        print(f"  Individual scores: {[f'{s:.5f}' for s in scores]}")
        if results['best_iterations']:
            print(f"  Average best iteration: {np.mean(results['best_iterations']):.1f}")
        print(f"{'='*60}")
    
    return results


def check_data_leakage(X_train, X_val, groups_train, groups_val, verbose=True):
    """
    检查数据泄露：验证训练集和验证集的用户是否完全分离 / Check data leakage: Verify if training and validation set users are completely separated
    
    Args:
        X_train: 训练集特征 / Training set features
        X_val: 验证集特征 / Validation set features
        groups_train: 训练集用户ID / Training set user IDs
        groups_val: 验证集用户ID / Validation set user IDs
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        bool: 如果无数据泄露返回True，否则返回False / Returns True if no data leakage, otherwise False
    """
    train_users = set(np.unique(groups_train))
    val_users = set(np.unique(groups_val))
    
    overlap = train_users & val_users
    
    if len(overlap) > 0:
        if verbose:
            print(f"⚠ Warning: Data leakage detected!")
            print(f"  Overlapping users: {len(overlap)}")
            print(f"  Overlapping user examples: {list(overlap)[:5]}")
        return False
    else:
        if verbose:
            print(f"✓ No data leakage: Training and validation set users are completely separated")
        return True


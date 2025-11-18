"""
超参数优化模块：使用Optuna进行贝叶斯优化（GPU加速） / Hyperparameter optimization module: Use Optuna for Bayesian optimization (GPU acceleration)
根据指南第三部分，使用TPE算法进行智能超参数搜索 / According to guide part 3, use TPE algorithm for intelligent hyperparameter search
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Warning: optuna not installed, cannot perform hyperparameter optimization")

# 导入baseline_model模块 / Import baseline_model module
try:
    from .baseline_model import detect_gpu_for_xgboost
except ImportError:
    # 如果相对导入失败，尝试绝对导入 / If relative import fails, try absolute import
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from baseline_model import detect_gpu_for_xgboost


def optimize_hyperparameters(X_train, y_train, groups, n_trials=100, 
                             cv_folds=5, use_gpu=True, timeout=None,
                             verbose=True):
    """
    使用Optuna进行XGBoost超参数优化 / Use Optuna for XGBoost hyperparameter optimization
    
    根据指南第三部分，搜索空间包括： / According to guide part 3, search space includes:
    - learning_rate: [1e-3, 0.1], log=True
    - max_depth: [3, 10]
    - subsample: [0.5, 1.0]
    - colsample_bytree: [0.5, 1.0]
    - min_child_weight: [1, 20]
    - gamma: [0.0, 5.0]
    
    Args:
        X_train: 训练特征DataFrame或numpy数组 / Training feature DataFrame or numpy array
        y_train: 训练标签Series或numpy数组 / Training label Series or numpy array
        groups: 用户ID数组（用于GroupKFold） / User ID array (for GroupKFold)
        n_trials: Optuna优化试验次数 / Number of Optuna optimization trials
        cv_folds: 交叉验证折数 / Number of cross-validation folds
        use_gpu: 是否使用GPU / Whether to use GPU
        timeout: 优化超时时间（秒），None表示不限制 / Optimization timeout (seconds), None means no limit
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        dict: 包含最佳参数和结果的字典 / Dictionary containing best parameters and results
            - 'best_params': 最佳参数字典 / Best parameter dictionary
            - 'best_score': 最佳CV分数 / Best CV score
            - 'study': Optuna study对象 / Optuna study object
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is not installed, cannot perform hyperparameter optimization. Please use 'pip install optuna' to install.")
    
    print("\n" + "="*60)
    print("Using Optuna for Hyperparameter Optimization (Bayesian Optimization)")
    print("="*60)
    print(f"  Number of optimization trials: {n_trials}")
    if timeout:
        print(f"  Timeout: {timeout} seconds")
    
    # 转换为numpy数组 / Convert to numpy arrays
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
    
    if isinstance(groups, pd.Series):
        groups_array = groups.values
    else:
        groups_array = groups
    
    # 检测GPU / Detect GPU
    if use_gpu:
        gpu_available, tree_method = detect_gpu_for_xgboost()
    else:
        gpu_available = False
        tree_method = 'hist'
    
    # 计算scale_pos_weight / Calculate scale_pos_weight
    negative_count = np.sum(y_train_array == 0)
    positive_count = np.sum(y_train_array == 1)
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.127
    
    # 导入GroupKFold / Import GroupKFold
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=cv_folds)
    
    def objective(trial):
        """
        Optuna优化目标函数 / Optuna optimization objective function
        返回GroupKFold交叉验证的平均准确率 / Returns average accuracy of GroupKFold cross-validation
        """
        # 定义搜索空间（根据指南表2） / Define search space (according to guide table 2)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'n_estimators': 2000,  # 固定值，使用early_stopping / Fixed value, use early_stopping
            'scale_pos_weight': scale_pos_weight,  # 固定值 / Fixed value
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': tree_method,  # GPU或CPU / GPU or CPU
        }
        
        # XGBoost 2.0+ 版本：如果GPU可用且tree_method是'hist'，添加device='cuda' / XGBoost 2.0+: If GPU available and tree_method is 'hist', add device='cuda'
        if gpu_available and tree_method == 'hist':
            params['device'] = 'cuda'
        
        # GroupKFold交叉验证 / GroupKFold cross-validation
        cv_scores = []
        best_iterations = []
        
        for train_idx, val_idx in gkf.split(X_train_array, y_train_array, groups=groups_array):
            X_fold_train = X_train_array[train_idx]
            X_fold_val = X_train_array[val_idx]
            y_fold_train = y_train_array[train_idx]
            y_fold_val = y_train_array[val_idx]
            
            # XGBoost 2.1.4+ 版本：early_stopping_rounds 应该在模型初始化时传递 / XGBoost 2.1.4+: early_stopping_rounds should be passed during model initialization
            # 而不是在 fit() 时传递 / Not during fit()
            model_params = params.copy()
            model_params['early_stopping_rounds'] = 50
            model = xgb.XGBClassifier(**model_params)
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(acc)
            
            if hasattr(model, 'best_iteration'):
                best_iterations.append(model.best_iteration)
        
        # 返回平均CV准确率 / Return average CV accuracy
        return np.mean(cv_scores)
    
    # 创建Optuna study / Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # 最大化准确率 / Maximize accuracy
        study_name='xgboost_hyperparameter_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE算法（贝叶斯优化） / TPE algorithm (Bayesian optimization)
    )
    
    # 运行优化 / Run optimization
    if verbose:
        print("\nStarting optimization...")
        print("  (This may take some time, especially in CPU mode)")
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # 提取最佳参数 / Extract best parameters
    best_params = study.best_params.copy()
    
    # 添加固定参数 / Add fixed parameters
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 2000,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': tree_method,
    })
    
    if verbose:
        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best cross-validation accuracy: {study.best_value:.5f} ({study.best_value*100:.2f}%)")
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}")
    
    return {
        'best_params': best_params,
        'best_score': study.best_value,
        'study': study
    }


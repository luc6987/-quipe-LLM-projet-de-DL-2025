"""
基线模型模块：实现指南第二部分的基线XGBoost模型 / Baseline model module: Implements baseline XGBoost model from guide part 2
关键参数：scale_pos_weight, early_stopping_rounds, GPU加速 / Key parameters: scale_pos_weight, early_stopping_rounds, GPU acceleration
"""

import os
import sys
# 添加 /Data 目录中的包路径（如果存在） / Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# GPU检测 / GPU detection
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU not available, will use CPU")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠ PyTorch not installed, cannot detect GPU")


def detect_gpu_for_xgboost():
    """
    检测XGBoost是否可以使用的GPU / Detect if XGBoost can use GPU
    
    XGBoost 2.0+ 版本使用 device='cuda' 和 tree_method='hist' 来启用GPU / XGBoost 2.0+ uses device='cuda' and tree_method='hist' to enable GPU
    旧版本的 tree_method='gpu_hist' 已被弃用 / Legacy tree_method='gpu_hist' is deprecated
    
    Returns:
        bool: GPU是否可用 / Whether GPU is available
        str: tree_method ('hist' for both CPU and GPU in 2.0+) / tree_method ('hist' for both CPU and GPU in 2.0+)
    """
    # 尝试检测XGBoost GPU支持 / Try to detect XGBoost GPU support
    try:
        # 创建一个小的测试数据集 / Create a small test dataset
        X_test = np.random.rand(100, 10)
        y_test = np.random.randint(0, 2, 100)
        
        # XGBoost 2.0+ 版本：使用 device='cuda' 和 tree_method='hist' / XGBoost 2.0+: Use device='cuda' and tree_method='hist'
        try:
            model_test = xgb.XGBClassifier(
                device='cuda',
                tree_method='hist',
                n_estimators=1,
                random_state=42
            )
            model_test.fit(X_test, y_test)
            # 如果成功，说明GPU可用 / If successful, GPU is available
            gpu_available = True
            tree_method = 'hist'
            print("✓ XGBoost GPU support available, will use device='cuda' and tree_method='hist'")
        except Exception as e1:
            # 尝试旧版本的 gpu_hist（向后兼容） / Try legacy gpu_hist (backward compatibility)
            try:
                model_test = xgb.XGBClassifier(
                    tree_method='gpu_hist',
                    n_estimators=1,
                    random_state=42
                )
                model_test.fit(X_test, y_test)
                gpu_available = True
                tree_method = 'gpu_hist'
                print("✓ XGBoost GPU support available (legacy API), will use tree_method='gpu_hist'")
            except Exception as e2:
                raise e1  # 抛出第一个异常（新API的异常） / Raise first exception (new API exception)
    except Exception as e:
        # GPU不可用，回退到CPU / GPU not available, fallback to CPU
        gpu_available = False
        tree_method = 'hist'
        print(f"⚠ XGBoost GPU not available: {e}")
        print("  Will use CPU mode (tree_method='hist')")
    
    return gpu_available, tree_method


def calculate_scale_pos_weight(y):
    """
    计算scale_pos_weight参数，用于处理类别不平衡 / Calculate scale_pos_weight parameter for handling class imbalance
    
    公式: scale_pos_weight = sum(negative_instances) / sum(positive_instances) / Formula: scale_pos_weight = sum(negative_instances) / sum(positive_instances)
    
    Args:
        y: 标签数组或Series / Label array or Series
        
    Returns:
        float: scale_pos_weight值 / scale_pos_weight value
    """
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = y
    
    negative_count = np.sum(y_array == 0)
    positive_count = np.sum(y_array == 1)
    
    if positive_count == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = negative_count / positive_count
    
    print(f"  Class distribution: Negative(0)={negative_count}, Positive(1)={positive_count}")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    
    return scale_pos_weight


def create_baseline_xgboost_params(y_train=None, use_gpu=True, **kwargs):
    """
    创建基线XGBoost参数字典
    
    根据指南第二部分，关键参数包括：
    - scale_pos_weight: 处理类别不平衡
    - early_stopping_rounds: 50（在fit时设置）
    - n_estimators: 2000
    - learning_rate: 0.05
    - eval_metric: 'logloss'
    - tree_method: 'gpu_hist' (GPU) 或 'hist' (CPU)
    
    Args:
        y_train: 训练标签（用于计算scale_pos_weight）
        use_gpu: 是否尝试使用GPU
        **kwargs: 额外的参数字典，会覆盖默认值
        
    Returns:
        dict: XGBoost参数字典
    """
    # 检测GPU
    if use_gpu:
        gpu_available, tree_method = detect_gpu_for_xgboost()
    else:
        gpu_available = False
        tree_method = 'hist'
    
    # 基础参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': tree_method,
    }
    
    # XGBoost 2.0+ 版本：如果GPU可用且tree_method是'hist'，添加device='cuda'
    if gpu_available and tree_method == 'hist':
        params['device'] = 'cuda'
    
    # 计算scale_pos_weight
    if y_train is not None:
        params['scale_pos_weight'] = calculate_scale_pos_weight(y_train)
    else:
        # 默认值（基于指南中的53%/47%分布）
        params['scale_pos_weight'] = 0.53 / 0.47  # ≈ 1.127
    
    # 应用用户提供的额外参数
    params.update(kwargs)
    
    return params


def train_baseline_model(X_train, y_train, groups=None, cv_folds=5, 
                        use_gpu=True, early_stopping_rounds=50, 
                        verbose=True):
    """
    训练基线XGBoost模型并进行GroupKFold交叉验证 / Train baseline XGBoost model and perform GroupKFold cross-validation
    
    Args:
        X_train: 训练特征DataFrame或numpy数组 / Training feature DataFrame or numpy array
        y_train: 训练标签Series或numpy数组 / Training label Series or numpy array
        groups: 用户ID数组（用于GroupKFold） / User ID array (for GroupKFold)
        cv_folds: 交叉验证折数 / Number of cross-validation folds
        use_gpu: 是否使用GPU / Whether to use GPU
        early_stopping_rounds: 早停轮数 / Early stopping rounds
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        dict: 包含模型和CV结果的字典 / Dictionary containing model and CV results
            - 'model': 最终训练的模型（在全部数据上） / Final trained model (on all data)
            - 'cv_results': GroupKFold CV结果 / GroupKFold CV results
            - 'params': 使用的参数字典 / Parameter dictionary used
    """
    print("\n" + "="*60)
    print("Training Baseline XGBoost Model")
    print("="*60)
    
    # 转换为numpy数组 / Convert to numpy arrays
    if hasattr(X_train, 'values'):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
    
    # 创建参数字典 / Create parameter dictionary
    params = create_baseline_xgboost_params(
        y_train=y_train_array,
        use_gpu=use_gpu
    )
    
    if verbose:
        print("\nXGBoost Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # GroupKFold交叉验证 / GroupKFold cross-validation
    from .validation import group_kfold_cv
    
    cv_results = group_kfold_cv(
        X=X_train_array,
        y=y_train_array,
        groups=groups,
        n_splits=cv_folds,
        model_params=params,
        eval_metric='accuracy',
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )
    
    # 计算平均最佳迭代次数（用于最终模型） / Calculate average best iteration (for final model)
    avg_best_iteration = None
    if cv_results['best_iterations']:
        avg_best_iteration = int(np.mean(cv_results['best_iterations']))
        # 设置为略高于平均值（因为使用更多数据） / Set to slightly above average (because using more data)
        final_n_estimators = int(avg_best_iteration * 1.1)
        params['n_estimators'] = final_n_estimators
        if verbose:
            print(f"\nAverage best iteration: {avg_best_iteration}")
            print(f"Final model n_estimators: {final_n_estimators}")
    
    # 在全部数据上训练最终模型 / Train final model on all data
    if verbose:
        print("\nTraining final model on all training data...")
    
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(
        X_train_array,
        y_train_array,
        verbose=False
    )
    
    if verbose:
        print("✓ Final model training complete")
    
    return {
        'model': final_model,
        'cv_results': cv_results,
        'params': params,
        'avg_best_iteration': avg_best_iteration
    }


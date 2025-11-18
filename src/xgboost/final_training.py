"""
最终训练和提交模块：在全部数据上训练最终模型并生成提交文件 / Final training and submission module: Train final model on all data and generate submission file
整合所有组件：特征工程、GroupKFold验证、超参数优化、特征选择 / Integrates all components: feature engineering, GroupKFold validation, hyperparameter optimization, feature selection
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 添加路径以便导入模块 / Add path for importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from baseline_model import create_baseline_xgboost_params, detect_gpu_for_xgboost
from validation import group_kfold_cv
from feature_selection import get_feature_importance, select_features_from_model
from hyperparameter_optimization import optimize_hyperparameters

try:
    from feature_engineering import engineer_features
except ImportError:
    pass


def train_final_model(X_train_full, y_train_full, groups=None, 
                     best_params=None, use_feature_selection=False,
                     feature_selection_threshold='median', use_gpu=True,
                     cv_folds=5, verbose=True):
    """
    在全部训练数据上训练最终模型 / Train final model on all training data
    
    根据指南第五部分： / According to guide part 5:
    - 使用Optuna找到的最佳超参数 / Use best hyperparameters found by Optuna
    - 使用特征选择后的特征子集（如果启用） / Use feature subset after feature selection (if enabled)
    - 计算最佳n_estimators（从CV中获取平均best_iteration） / Calculate best n_estimators (get average best_iteration from CV)
    - 使用GPU加速 / Use GPU acceleration
    
    Args:
        X_train_full: 全部训练特征 / All training features
        y_train_full: 全部训练标签 / All training labels
        groups: 用户ID数组（用于GroupKFold） / User ID array (for GroupKFold)
        best_params: 最佳超参数字典（如果为None，使用基线参数） / Best hyperparameter dictionary (if None, use baseline params)
        use_feature_selection: 是否使用特征选择 / Whether to use feature selection
        feature_selection_threshold: 特征选择阈值 / Feature selection threshold
        use_gpu: 是否使用GPU / Whether to use GPU
        cv_folds: 交叉验证折数（用于确定最佳n_estimators）
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含最终模型和信息的字典
    """
    print("\n" + "="*60)
    print("Final Model Training")
    print("="*60)
    
    # 转换为numpy数组
    if hasattr(X_train_full, 'values'):
        X_train_array = X_train_full.values
        feature_names = X_train_full.columns.tolist()
    else:
        X_train_array = X_train_full
        feature_names = [f'feature_{i}' for i in range(X_train_array.shape[1])]
    
    if hasattr(y_train_full, 'values'):
        y_train_array = y_train_full.values
    else:
        y_train_array = y_train_full
    
    # 检测GPU
    if use_gpu:
        gpu_available, tree_method = detect_gpu_for_xgboost()
    else:
        gpu_available = False
        tree_method = 'hist'
    
    # 确定最终参数
    if best_params is None:
        # 使用基线参数
        final_params = create_baseline_xgboost_params(
            y_train=y_train_array,
            use_gpu=use_gpu
        )
    else:
        final_params = best_params.copy()
        # 确保tree_method正确
        final_params['tree_method'] = tree_method
    
    # 特征选择（如果启用） / Feature selection (if enabled)
    selected_features_mask = None
    if use_feature_selection:
        print("\nPerforming feature selection...")
        
        # 先训练一个临时模型用于特征选择 / First train a temporary model for feature selection
        temp_model = xgb.XGBClassifier(**final_params)
        temp_model.fit(X_train_array, y_train_array, verbose=False)
        
        # 使用SelectFromModel选择特征 / Use SelectFromModel to select features
        selector, X_train_selected = select_features_from_model(
            temp_model,
            pd.DataFrame(X_train_array, columns=feature_names),
            pd.Series(y_train_array),
            threshold=feature_selection_threshold,
            prefit=True
        )
        
        selected_features_mask = selector.get_support()
        X_train_array = X_train_selected.values
        feature_names = X_train_selected.columns.tolist()
        
        print(f"  Selected features: {len(feature_names)}/{len(selected_features_mask)}")
    
    # 确定最佳n_estimators（通过GroupKFold CV） / Determine best n_estimators (via GroupKFold CV)
    print("\nDetermining best n_estimators via GroupKFold CV...")
    
    # 临时参数（用于CV）
    cv_params = final_params.copy()
    cv_params['n_estimators'] = 2000  # 足够大的值，使用early_stopping
    
    cv_results = group_kfold_cv(
        X=X_train_array,
        y=y_train_array,
        groups=groups,
        n_splits=cv_folds,
        model_params=cv_params,
        eval_metric='accuracy',
        early_stopping_rounds=50,
        verbose=verbose
    )
    
    # 计算平均最佳迭代次数 / Calculate average best iteration
    if cv_results['best_iterations']:
        avg_best_iteration = int(np.mean(cv_results['best_iterations']))
        # 设置为略高于平均值（因为使用更多数据） / Set to slightly above average (because using more data)
        final_n_estimators = int(avg_best_iteration * 1.1)
    else:
        # 如果没有best_iterations，使用默认值 / If no best_iterations, use default value
        final_n_estimators = final_params.get('n_estimators', 2000)
        avg_best_iteration = None
    
    final_params['n_estimators'] = final_n_estimators
    
    if verbose:
        if avg_best_iteration:
            print(f"  Average best iteration: {avg_best_iteration}")
        print(f"  Final n_estimators: {final_n_estimators}")
        print(f"  CV mean accuracy: {cv_results['mean_score']:.5f}")
    
    # 在全部数据上训练最终模型 / Train final model on all training data
    print("\nTraining final model on all training data...")
    
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        X_train_array,
        y_train_array,
        verbose=verbose
    )
    
    if verbose:
        print("✓ Final model training completed")
    
    return {
        'model': final_model,
        'params': final_params,
        'feature_names': feature_names,
        'selected_features_mask': selected_features_mask,
        'cv_results': cv_results,
        'avg_best_iteration': avg_best_iteration
    }


def generate_submission(model, X_test, test_ids, feature_names=None,
                       output_file='output/submission.csv', verbose=True):
    """
    生成提交文件 / Generate submission file
    
    Args:
        model: 训练好的模型 / Trained model
        X_test: 测试特征 / Test features
        test_ids: 测试集ID / Test set IDs
        feature_names: 特征名称列表（用于对齐） / Feature name list (for alignment)
        output_file: 输出文件路径 / Output file path
        verbose: 是否打印详细信息 / Whether to print detailed information
        
    Returns:
        DataFrame: 提交DataFrame / Submission DataFrame
    """
    print("\n" + "="*60)
    print("Generating Submission File")
    print("="*60)
    
    # 确保测试集特征与训练集一致 / Ensure test set features match training set
    if feature_names is not None and hasattr(X_test, 'columns'):
        # 对齐特征列 / Align feature columns
        missing_cols = set(feature_names) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(feature_names)
        
        if missing_cols:
            if verbose:
                print(f"  Adding missing columns: {len(missing_cols)} (filled with 0)")
            for col in missing_cols:
                X_test[col] = 0
        
        if extra_cols:
            if verbose:
                print(f"  Removing extra columns: {len(extra_cols)}")
            X_test = X_test.drop(columns=list(extra_cols))
        
        # 确保列顺序一致 / Ensure column order is consistent
        X_test = X_test[feature_names]
    
    # 转换为numpy数组 / Convert to numpy array
    if hasattr(X_test, 'values'):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    # 预测 / Prediction
    predictions = model.predict(X_test_array)
    predictions_proba = model.predict_proba(X_test_array)[:, 1]
    
    # 创建提交DataFrame / Create submission DataFrame
    submission = pd.DataFrame({
        'ID': test_ids,
        'Prediction': predictions
    })
    
    # 确保输出目录存在 / Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 保存提交文件 / Save submission file
    submission.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n✓ Submission file saved to: {output_file}")
        print(f"  Prediction samples: {len(predictions)}")
        print(f"  Prediction value distribution:")
        print(submission['Prediction'].value_counts().sort_index())
        
        # 健全性检查 / Sanity check
        pred_ratio = predictions.sum() / len(predictions)
        if pred_ratio < 0.3 or pred_ratio > 0.7:
            print(f"\n⚠ Warning: Prediction distribution may be abnormal (Influencer ratio: {pred_ratio:.2%})")
            print("  Expected range: 40%-50%")
        else:
            print(f"\n✓ Prediction distribution normal (Influencer ratio: {pred_ratio:.2%})")
    
    return submission


def main():
    """
    主函数：完整的最终训练和提交流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Final model training and submission')
    parser.add_argument('--train-features', type=str, default='cache/cleaned_features.csv',
                       help='Training feature file path')
    parser.add_argument('--train-labels', type=str, default='cache/train_labels.csv',
                       help='Training label file path')
    parser.add_argument('--train-user-ids', type=str, default='cache/train_user_ids.csv',
                       help='Training user ID file path (for GroupKFold)')
    parser.add_argument('--test-features', type=str, default='cache/test_cleaned_features.csv',
                       help='Test feature file path')
    parser.add_argument('--test-ids', type=str, default='cache/test_ids.csv',
                       help='Test ID file path')
    parser.add_argument('--output', type=str, default='output/submission.csv',
                       help='Output submission file path')
    parser.add_argument('--use-feature-selection', action='store_true',
                       help='Use feature selection')
    parser.add_argument('--feature-selection-threshold', type=str, default='median',
                       help='Feature selection threshold')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU acceleration')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Final Model Training and Submission")
    print("="*60)
    
    # 加载数据 / Load data
    print("\nLoading data...")
    X_train = pd.read_csv(args.train_features)
    y_train = pd.read_csv(args.train_labels)['label']
    
    # 加载用户ID / Load user IDs
    if os.path.exists(args.train_user_ids):
        groups = pd.read_csv(args.train_user_ids)['user_id'].values
        print(f"  Loaded user IDs: {len(np.unique(groups))} unique users")
    else:
        groups = None
        print("  ⚠ Warning: User ID file not found, cannot use GroupKFold")
    
    # 加载测试数据 / Load test data
    X_test = pd.read_csv(args.test_features)
    test_ids = pd.read_csv(args.test_ids)['challenge_id']
    
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # 训练最终模型
    result = train_final_model(
        X_train_full=X_train,
        y_train_full=y_train,
        groups=groups,
        best_params=None,  # 可以使用Optuna优化的参数
        use_feature_selection=args.use_feature_selection,
        feature_selection_threshold=args.feature_selection_threshold,
        use_gpu=args.use_gpu,
        cv_folds=args.cv_folds,
        verbose=True
    )
    
    # 生成提交文件
    submission = generate_submission(
        model=result['model'],
        X_test=X_test,
        test_ids=test_ids,
        feature_names=result['feature_names'],
        output_file=args.output,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Completed!")
    print("="*60)
    print(f"Submission file: {args.output}")
    print(f"CV mean accuracy: {result['cv_results']['mean_score']:.5f}")


if __name__ == "__main__":
    main()


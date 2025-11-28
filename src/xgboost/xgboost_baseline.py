"""
策略一：利用194个表格特征建立强大的基线 (GBDT) / Strategy 1: Build a Strong Baseline Using 194 Tabular Features (GBDT)
使用XGBoost进行梯度提升决策树建模 / Use XGBoost for Gradient Boosting Decision Tree Modeling

Updated to use shared utils.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings

# Add project root to path to allow imports from src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import data_processing as dp

warnings.filterwarnings('ignore')


def get_default_params():
    """
    获取XGBoost默认参数 / Get default XGBoost parameters
    """
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }


def train_single_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    训练单个XGBoost模型 / Train a single XGBoost model
    """
    if params is None:
        params = get_default_params()
    
    model = xgb.XGBClassifier(**params)
    
    eval_set = []
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
        
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    return model


def train_xgboost_model(X_train, y_train, cv_folds=5):
    """
    训练XGBoost模型并进行交叉验证 / Train XGBoost model and perform cross-validation
    """
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    
    # 转换为numpy数组 / Convert to numpy arrays
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    # 设置5折分层交叉验证 / Set up 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # XGBoost参数 / XGBoost parameters
    xgb_params = get_default_params()
    
    print(f"\nXGBoost Parameters:")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    # 交叉验证 / Cross-validation
    print(f"\nRunning {cv_folds}-Fold Stratified Cross-Validation...")
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train_array), 1):
        print(f"\nFold {fold}/{cv_folds}:")
        
        X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
        y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
        
        # 使用提取出的函数训练 / Train using extracted function
        model = train_single_model(
            X_fold_train, y_fold_train, 
            X_fold_val, y_fold_val, 
            params=xgb_params
        )
        
        # 预测和评估 / Prediction and evaluation
        y_pred = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, y_pred)
        cv_scores.append(acc)
        models.append(model)
        
        print(f"  Validation Accuracy: {acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores)*100:.2f}%)")
    print(f"  Std Deviation: {np.std(cv_scores):.4f} ({np.std(cv_scores)*100:.2f}%)")
    print(f"  Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"{'='*60}")
    
    # 训练最终模型（使用全部训练数据） / Train final model (using all training data)
    print("\nTraining final model on all training data...")
    final_model = train_single_model(X_train_array, y_train_array, params=xgb_params)
    print("Training complete.")
    
    return final_model, cv_scores, models



def make_predictions(model, X_test, test_ids):
    """
    生成预测结果 / Generate predictions
    """
    print("\nMaking predictions on test data...")
    X_test_array = X_test.values
    predictions = model.predict(X_test_array)
    
    # 创建提交文件 / Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'Prediction': predictions
    })
    
    return submission


def main():
    """
    主函数：执行完整的训练和预测流程 / Main function: Execute complete training and prediction pipeline
    """
    print("="*60)
    print("Strategy 1: Build a Strong Baseline Using 194 Tabular Features (GBDT)")
    print("="*60)
    
    # Use unified path configuration
    from src.utils.paths import DATA_DIR
    
    # 1. 加载数据 / 1. Load data
    train_df = dp.load_data(str(DATA_DIR / 'train.jsonl'), clean_text=True)
    test_df = dp.load_data(str(DATA_DIR / 'kaggle_test.jsonl'), clean_text=True)
    
    # 2. 特征工程 / Feature Engineering
    # We process train and test separately but pass transformers
    
    # Process Train
    print("\nProcessing Training Data...")
    train_engineered, transformers = dp.engineer_features(train_df, is_train=True)
    
    # Process Test
    print("\nProcessing Test Data...")
    test_engineered, _ = dp.engineer_features(test_df, is_train=False, transformers=transformers)
    
    # 3. Prepare X, y
    X_train, y_train, _ = dp.prepare_xy(train_engineered, target_col='label')
    X_test, _, test_ids = dp.prepare_xy(test_engineered, target_col='label') # label col might not exist in test, but prepare_xy handles it if it's not there
    
    # Ensure columns match
    # Note: prepare_xy drops target and id. 
    # We need to ensure X_test has same columns as X_train
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    # Add missing cols to test
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    for c in missing_in_test:
        X_test[c] = 0 # Default fill
    
    # Drop extra cols in test
    X_test = X_test[X_train.columns]
    
    print(f"\nFinal feature matrix shape:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 4. 训练模型 / 4. Train model
    model, cv_scores, _ = train_xgboost_model(X_train, y_train, cv_folds=5)
    
    # 5. 生成预测 / 5. Generate predictions
    submission = make_predictions(model, X_test, test_ids)
    
    # 6. 保存结果 / 6. Save results
    from src.utils.paths import get_submissions_dir
    output_file = str(get_submissions_dir() / 'xgboost_baseline.csv')
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # 7. 特征重要性 / 7. Feature importance
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*60)
    print("Strategy 1 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

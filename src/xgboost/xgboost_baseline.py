"""
策略一：利用194个表格特征建立强大的基线 (GBDT)
使用XGBoost进行梯度提升决策树建模
"""

import json
import os
import numpy as np
import pandas as pd
from pandas import json_normalize
import warnings
warnings.filterwarnings('ignore')

# XGBoost
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def load_data(train_file, test_file=None):
    """
    加载训练和测试数据，将嵌套JSON展平为表格格式
    使用逐行处理以避免内存问题
    """
    print("Loading training data...")
    # 逐行读取并展平，更节省内存
    records = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} lines...")
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"  Loaded {len(records)} records, normalizing...")
    train_data = json_normalize(records)
    print(f"Training data shape: {train_data.shape}")
    
    if test_file:
        print("Loading test data...")
        records = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    records.append(data)
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {len(records)} records, normalizing...")
        test_data = json_normalize(records)
        print(f"Test data shape: {test_data.shape}")
        return train_data, test_data
    
    return train_data


def extract_table_features(df, is_train=True):
    """
    分离表格特征和文本特征
    保留所有非文本列作为表格特征（194个特征）
    """
    # 需要排除的列
    exclude_cols = ['text', 'extended_tweet.full_text', 'challenge_id']
    if is_train:
        exclude_cols.append('label')
    
    # 获取所有表格特征列
    table_feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 提取表格特征
    X_table = df[table_feature_cols].copy()
    
    print(f"Extracted {len(table_feature_cols)} table features")
    print(f"Table features shape: {X_table.shape}")
    
    return X_table, table_feature_cols


def calculate_account_age(created_at_str):
    """
    计算账户年龄（天数）
    """
    if pd.isna(created_at_str):
        return None
    
    try:
        # Twitter created_at格式: "Wed Mar 17 12:59:53 +0000 2021"
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        now = datetime.now(created_at.tzinfo)
        age_days = (now - created_at).days
        return age_days
    except:
        return None


def engineer_features(X_table, user_cols=None):
    """
    特征工程：创建关键比率特征
    根据竞赛定义，创建最重要的特征
    """
    print("\nEngineering features...")
    X_engineered = X_table.copy()
    
    # 1. follower_following_ratio - 黄金特征
    # 查找用户相关的列（优先使用主推文的用户特征）
    if user_cols is None:
        # 尝试自动识别用户特征列
        # 优先使用主推文的用户特征（user.followers_count），而不是quoted_status.user.followers_count
        user_followers_cols = [col for col in X_table.columns if 'followers_count' in col.lower() and 'quoted_status' not in col.lower()]
        if not user_followers_cols:
            user_followers_cols = [col for col in X_table.columns if 'followers_count' in col.lower()]
        
        user_friends_cols = [col for col in X_table.columns if 'friends_count' in col.lower() and 'quoted_status' not in col.lower()]
        if not user_friends_cols:
            user_friends_cols = [col for col in X_table.columns if 'friends_count' in col.lower()]
        
        user_statuses_cols = [col for col in X_table.columns if 'statuses_count' in col.lower() and 'quoted_status' not in col.lower()]
        if not user_statuses_cols:
            user_statuses_cols = [col for col in X_table.columns if 'statuses_count' in col.lower()]
        
        user_created_cols = [col for col in X_table.columns if 'user.created_at' in col.lower() and 'quoted_status' not in col.lower()]
        if not user_created_cols:
            user_created_cols = [col for col in X_table.columns if 'created_at' in col.lower() and 'user' in col.lower()]
    else:
        user_followers_cols = [col for col in user_cols if 'followers_count' in col.lower()]
        user_friends_cols = [col for col in user_cols if 'friends_count' in col.lower()]
        user_statuses_cols = [col for col in user_cols if 'statuses_count' in col.lower()]
        user_created_cols = [col for col in user_cols if 'created_at' in col.lower()]
    
    # 创建 follower_following_ratio
    if user_followers_cols and user_friends_cols:
        followers_col = user_followers_cols[0]
        friends_col = user_friends_cols[0]
        # 只对非缺失值计算比率
        mask = X_engineered[followers_col].notna() & X_engineered[friends_col].notna()
        X_engineered['follower_following_ratio'] = 0.0
        X_engineered.loc[mask, 'follower_following_ratio'] = (
            X_engineered.loc[mask, followers_col] / 
            (X_engineered.loc[mask, friends_col] + 1)
        )
        print(f"  Created: follower_following_ratio (using {followers_col} / {friends_col})")
        print(f"    Non-null ratio values: {mask.sum()}/{len(X_engineered)}")
    
    # 2. user_activity_rate
    if user_statuses_cols and user_created_cols:
        statuses_col = user_statuses_cols[0]
        created_col = user_created_cols[0]
        
        # 计算账户年龄
        account_ages = X_engineered[created_col].apply(calculate_account_age)
        X_engineered['account_age_days'] = account_ages
        
        # 计算活动率（只对有效值计算）
        mask = X_engineered[statuses_col].notna() & X_engineered['account_age_days'].notna()
        X_engineered['user_activity_rate'] = 0.0
        X_engineered.loc[mask, 'user_activity_rate'] = (
            X_engineered.loc[mask, statuses_col] / 
            (X_engineered.loc[mask, 'account_age_days'] + 1)
        )
        print(f"  Created: user_activity_rate (using {statuses_col} / account_age_days)")
        print(f"    Non-null activity rate values: {mask.sum()}/{len(X_engineered)}")
    
    # 3. tweet_engagement_rate
    retweet_cols = [col for col in X_table.columns if 'retweet_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not retweet_cols:
        retweet_cols = [col for col in X_table.columns if 'retweet_count' in col.lower()]
    
    favorite_cols = [col for col in X_table.columns if 'favorite_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not favorite_cols:
        favorite_cols = [col for col in X_table.columns if 'favorite_count' in col.lower()]
    
    if retweet_cols and favorite_cols and user_followers_cols:
        retweet_col = retweet_cols[0]
        favorite_col = favorite_cols[0]
        followers_col = user_followers_cols[0]
        
        # 只对有效值计算参与率
        mask = (X_engineered[retweet_col].notna() & 
                X_engineered[favorite_col].notna() & 
                X_engineered[followers_col].notna())
        X_engineered['tweet_engagement_rate'] = 0.0
        X_engineered.loc[mask, 'tweet_engagement_rate'] = (
            (X_engineered.loc[mask, retweet_col] + 
             X_engineered.loc[mask, favorite_col]) / 
            (X_engineered.loc[mask, followers_col] + 1)
        )
        print(f"  Created: tweet_engagement_rate")
        print(f"    Non-null engagement rate values: {mask.sum()}/{len(X_engineered)}")
    
    return X_engineered


def handle_missing_values(X_train, X_test=None):
    """
    处理缺失值：使用中位数填充数值特征，0填充其他
    """
    print("\nHandling missing values...")
    
    # 分离数值列和非数值列
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 对数值列使用中位数填充
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(median_val)
    
    # 对非数值列使用0或空字符串填充
    for col in non_numeric_cols:
        X_train[col] = X_train[col].fillna(0)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(0)
    
    print(f"  Filled {len(numeric_cols)} numeric columns with median")
    print(f"  Filled {len(non_numeric_cols)} non-numeric columns with 0")
    
    return X_train, X_test


def encode_categorical_features(X_train, X_test=None):
    """
    编码分类特征：将非数值列转换为数值
    """
    print("\nEncoding categorical features...")
    
    # 识别分类列
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # 合并训练和测试数据以获取所有可能的类别
        if X_test is not None:
            combined = pd.concat([X_train[col], X_test[col]], axis=0)
            le.fit(combined.astype(str))
        else:
            le.fit(X_train[col].astype(str))
        
        X_train[col] = le.transform(X_train[col].astype(str))
        if X_test is not None:
            X_test[col] = le.transform(X_test[col].astype(str))
        
        label_encoders[col] = le
    
    print(f"  Encoded {len(categorical_cols)} categorical columns")
    
    return X_train, X_test, label_encoders


def train_xgboost_model(X_train, y_train, cv_folds=5):
    """
    训练XGBoost模型并进行交叉验证
    """
    print("\n" + "="*60)
    print("Training XGBoost Model")
    print("="*60)
    
    # 转换为numpy数组
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    # 设置5折分层交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # XGBoost参数
    xgb_params = {
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
        'tree_method': 'hist'  # 更快的训练方法
    }
    
    print(f"\nXGBoost Parameters:")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    # 交叉验证
    print(f"\nRunning {cv_folds}-Fold Stratified Cross-Validation...")
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train_array), 1):
        print(f"\nFold {fold}/{cv_folds}:")
        
        X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
        y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
        
        # 创建XGBoost模型
        model = xgb.XGBClassifier(**xgb_params)
        
        # 训练
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        # 预测和评估
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
    
    # 训练最终模型（使用全部训练数据）
    print("\nTraining final model on all training data...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_train_array, y_train_array)
    print("Training complete.")
    
    return final_model, cv_scores, models


def make_predictions(model, X_test, test_ids):
    """
    生成预测结果
    """
    print("\nMaking predictions on test data...")
    X_test_array = X_test.values
    predictions = model.predict(X_test_array)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'ID': test_ids,
        'Prediction': predictions
    })
    
    return submission


def main():
    """
    主函数：执行完整的训练和预测流程
    """
    print("="*60)
    print("策略一：利用194个表格特征建立强大的基线 (GBDT)")
    print("="*60)
    
    # Get the project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'data')
    
    # 1. 加载数据
    train_data, test_data = load_data(
        os.path.join(data_dir, 'train.jsonl'), 
        os.path.join(data_dir, 'kaggle_test.jsonl')
    )
    
    # 2. 分离特征和标签
    y_train = train_data['label']
    test_ids = test_data['challenge_id']
    
    # 3. 提取表格特征（排除文本）
    X_train_table, feature_cols = extract_table_features(train_data, is_train=True)
    X_test_table, _ = extract_table_features(test_data, is_train=False)
    
    # 确保测试集有相同的列
    # 添加缺失的列（用0填充）
    missing_cols = set(X_train_table.columns) - set(X_test_table.columns)
    for col in missing_cols:
        X_test_table[col] = 0
    
    # 确保列顺序一致
    X_test_table = X_test_table[X_train_table.columns]
    
    # 4. 特征工程
    X_train_engineered = engineer_features(X_train_table)
    X_test_engineered = engineer_features(X_test_table)
    
    # 确保测试集有相同的工程特征列
    missing_eng_cols = set(X_train_engineered.columns) - set(X_test_engineered.columns)
    for col in missing_eng_cols:
        X_test_engineered[col] = 0
    
    X_test_engineered = X_test_engineered[X_train_engineered.columns]
    
    # 5. 处理缺失值
    X_train_processed, X_test_processed = handle_missing_values(
        X_train_engineered, X_test_engineered
    )
    
    # 6. 编码分类特征
    X_train_final, X_test_final, _ = encode_categorical_features(
        X_train_processed, X_test_processed
    )
    
    print(f"\nFinal feature matrix shape:")
    print(f"  Training: {X_train_final.shape}")
    print(f"  Test: {X_test_final.shape}")
    
    # 7. 训练模型
    model, cv_scores, _ = train_xgboost_model(X_train_final, y_train, cv_folds=5)
    
    # 8. 生成预测
    submission = make_predictions(model, X_test_final, test_ids)
    
    # 9. 保存结果
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'xgboost_baseline.csv')
    submission.to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # 10. 特征重要性
    print("\n" + "="*60)
    print("Top 20 Most Important Features:")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*60)
    print("Strategy 1 Complete!")
    print("="*60)


if __name__ == "__main__":
    main()


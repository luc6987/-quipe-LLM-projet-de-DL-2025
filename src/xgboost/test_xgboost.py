"""
XGBoost / Quick test XGBoost baseline model
"""
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Quick Test: XGBoost Baseline Model")
print("="*60)

# 1. 10000 / Load data (use first 10000 rows for quick test)
print("\n1. Loading data (first 10000 rows for quick test)...")
records = []
with open('train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        try:
            data = json.loads(line.strip())
            records.append(data)
        except:
            continue

print(f"   Loaded {len(records)} records")
train_data = json_normalize(records)
print(f"   Shape: {train_data.shape}")

# 2.  / Separate features and labels
y_train = train_data['label']
print(f"\n2. Labels distribution:")
print(f"   Class 0: {(y_train == 0).sum()}")
print(f"   Class 1: {(y_train == 1).sum()}")

# 3.  / Extract tabular features (exclude text)
exclude_cols = ['text', 'extended_tweet.full_text', 'label']
table_cols = [col for col in train_data.columns if col not in exclude_cols]
X_train = train_data[table_cols].copy()
print(f"\n3. Extracted {len(table_cols)} table features")

# 4.  / Handle missing values
print("\n4. Handling missing values...")
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    X_train[col] = X_train[col].fillna(X_train[col].median())

non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    X_train[col] = X_train[col].fillna(0)

# 5.  / Encode categorical features
print("5. Encoding categorical features...")
le_dict = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    le_dict[col] = le

# 6. numpy / Convert to numpy arrays
X_array = X_train.values
y_array = y_train.values
print(f"\n6. Final feature matrix shape: {X_array.shape}")

# 7. 5 / 5-fold cross-validation
print("\n7. Running 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,  #  / Reduce number of trees to speed up test
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), 1):
    print(f"\n   Fold {fold}/5:")
    X_fold_train, X_fold_val = X_array[train_idx], X_array[val_idx]
    y_fold_train, y_fold_val = y_array[train_idx], y_array[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_fold_train, y_fold_train, verbose=False)
    
    y_pred = model.predict(X_fold_val)
    acc = accuracy_score(y_fold_val, y_pred)
    cv_scores.append(acc)
    print(f"     Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print(f"\n{'='*60}")
print(f"Cross-Validation Results:")
print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores)*100:.2f}%)")
print(f"  Std Deviation: {np.std(cv_scores):.4f}")
print(f"{'='*60}")

print("\n✓ Quick test completed successfully!")
print("  You can now run the full xgboost_baseline.py script on all data.")






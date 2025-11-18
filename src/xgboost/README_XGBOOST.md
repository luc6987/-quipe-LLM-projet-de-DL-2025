# Strategy 1: Build a Strong Baseline Using 194 Tabular Features (GBDT)

## Overview

This implementation uses XGBoost gradient boosting decision tree model, focusing on leveraging 194 tabular features (metadata) to build a strong baseline model.

## Core Features

1. **Tabular Features Only**: Exclude text features, focus on metadata (194 features)
2. **Feature Engineering**: Create key ratio features
   - `follower_following_ratio`: Follower/following ratio (golden feature)
   - `user_activity_rate`: User activity rate
   - `tweet_engagement_rate`: Tweet engagement rate
3. **5-Fold Stratified Cross-Validation**: Ensure reliable model evaluation
4. **Complete Preprocessing Pipeline**: Missing value handling, categorical feature encoding

## File Description

- `xgboost_baseline.py`: Complete implementation, processes all training data
- `test_xgboost.py`: Quick test version (uses first 10000 rows of data)

## Quick Test

Run the quick test version (approximately 1-2 minutes):

```bash
python3 test_xgboost.py
```

Expected result: Approximately 84-85% cross-validation accuracy

## Full Training

Run the complete version (processes all 154k training data, may take 10-30 minutes):

```bash
python3 xgboost_baseline.py
```

Output file: `xgboost_baseline.csv`

## Model Parameters

Current XGBoost parameters used:

- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 3
- `gamma`: 0.1
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0

## Performance Comparison

- **Logistic Regression Baseline** (using text features): ~62.57%
- **XGBoost Baseline** (using 194 tabular features): ~84-85%

This demonstrates the importance of metadata features in this problem.

## Next Steps for Optimization

1. Hyperparameter tuning (using Optuna or GridSearch)
2. Add more feature engineering
3. Try LightGBM as an alternative
4. Ensemble multiple models

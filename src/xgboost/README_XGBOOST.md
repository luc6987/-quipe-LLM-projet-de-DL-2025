# 策略一：利用194个表格特征建立强大的基线 (GBDT)

## 概述

本实现使用XGBoost梯度提升决策树模型，专注于利用194个表格特征（元数据）来建立强大的基线模型。

## 核心特点

1. **只使用表格特征**：排除文本特征，专注于元数据（194个特征）
2. **特征工程**：创建关键比率特征
   - `follower_following_ratio`: 粉丝/关注比率（黄金特征）
   - `user_activity_rate`: 用户活动率
   - `tweet_engagement_rate`: 推文参与率
3. **5折分层交叉验证**：确保模型评估的可靠性
4. **完整的预处理流程**：缺失值处理、分类特征编码

## 文件说明

- `xgboost_baseline.py`: 完整实现，处理全部训练数据
- `test_xgboost.py`: 快速测试版本（使用前10000行数据）

## 快速测试

运行快速测试版本（约1-2分钟）：

```bash
python3 test_xgboost.py
```

预期结果：约84-85%的交叉验证准确率

## 完整训练

运行完整版本（处理全部154k训练数据，可能需要10-30分钟）：

```bash
python3 xgboost_baseline.py
```

输出文件：`xgboost_baseline.csv`

## 模型参数

当前使用的XGBoost参数：

- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 3
- `gamma`: 0.1
- `reg_alpha`: 0.1
- `reg_lambda`: 1.0

## 性能对比

- **Logistic Regression基线**（使用文本特征）: ~62.57%
- **XGBoost基线**（使用194个表格特征）: ~84-85%

这证明了元数据特征在此问题中的重要性。

## 下一步优化

1. 超参数调优（使用Optuna或GridSearch）
2. 添加更多特征工程
3. 尝试LightGBM作为替代
4. 集成多个模型






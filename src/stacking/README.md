# Stacking Ensemble Module

## 概述 / Overview

这个模块实现了三个boosting模型的堆叠集成：LightGBM + XGBoost + CatBoost。

This module implements stacking ensemble for three boosting models: LightGBM + XGBoost + CatBoost.

## 使用方法 / Usage

### 方法1: 通过main.py和setup.yml控制

1. 在 `setup.yml` 中启用stacking模式：
```yaml
stacking:
  enabled: true  # 设置为true启用stacking
  n_folds: 5
  text_components: 128
```

2. 运行主脚本：
```bash
python src/main.py
```

### 方法2: 直接运行stacking模块

```bash
python src/stacking/main.py
```

或者：

```bash
python -m src.stacking.main
```

## 配置选项 / Configuration

在 `setup.yml` 的 `stacking` 部分可以配置：

- `enabled`: 是否启用stacking模式（默认: false）
- `n_folds`: 交叉验证折数（默认: 5）
- `text_components`: 文本嵌入PCA降维后的维度（默认: 128，当前未使用）

## 输出文件 / Output Files

所有输出文件保存在 `output/stacking/` 目录：

- `correlations_oof.csv`: OOF预测的相关性矩阵
- `correlations_test.csv`: 测试集预测的相关性矩阵
- `oof_predictions.npz`: 所有模型的OOF logits
- `test_predictions.npz`: 所有模型的测试集logits
- `meta_model_info.json`: 元模型信息和性能指标

提交文件保存在 `output/submissions/submission_stacking_boosting.csv`

## 工作流程 / Workflow

1. **生成OOF预测**: 对每个模型（LightGBM, XGBoost, CatBoost）进行5折交叉验证，生成OOF logits
2. **计算相关性**: 计算模型预测之间的Pearson相关系数
3. **训练元模型**: 使用三个模型的OOF logits作为特征，训练Logistic Regression元模型
4. **生成预测**: 使用元模型对测试集进行预测并生成提交文件

## 注意事项 / Notes

- Stacking模式会训练三个完整的模型，因此运行时间较长
- 确保在 `setup.yml` 中正确配置了各个模型的参数
- GPU加速会自动检测和使用（如果可用）


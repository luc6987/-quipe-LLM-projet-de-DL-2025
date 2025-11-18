# Complete XGBoost Solution for Influencer/Observer Classification

## Overview

This is a **complete, standalone solution** for the Kaggle "Influencer or Observer" classification challenge. The solution uses **only metadata features** (no text content) and implements a production-ready pipeline with GPU acceleration, Bayesian hyperparameter optimization, and feature interpretability analysis.

## Key Features

1. **Metadata-Only Features**: Strictly follows competition rules - NO text content used
2. **GPU Acceleration**: All XGBoost models use GPU acceleration for faster training
3. **GroupKFold Validation**: Prevents data leakage by ensuring same-user tweets stay together
4. **Bayesian Optimization**: Uses Optuna with TPE algorithm for efficient hyperparameter search
5. **SHAP Analysis**: Provides feature interpretability and importance rankings

## Theoretical Explanations

### Why XGBoost?

**Etymology**: "XGBoost" = eXtreme Gradient Boosting

**Mathematical Foundation**:
- **Gradient Boosting**: Iteratively adds weak learners (trees) that minimize residual errors
- **Second-Order Derivatives**: XGBoost uses second-order Taylor expansion for more accurate loss approximation
- **Regularization**: L1 (alpha) and L2 (lambda) regularization prevent overfitting
- **Efficiency**: Histogram-based algorithm for fast tree construction

**Why Tree Models?**
- Handle non-linear relationships naturally
- Feature interactions captured automatically
- Robust to outliers and missing values
- No need for feature scaling

### Why Optuna?

**Bayesian Optimization Principle**:
- Builds probabilistic model of objective function: p(f(x) | observations)
- Updates posterior distribution after each trial: p(f(x) | new observation)
- Samples from acquisition function (e.g., Expected Improvement) to balance exploration/exploitation

**TPE Algorithm** (Tree-structured Parzen Estimator):
- Models p(x|good) and p(x|bad) distributions separately
- Samples from p(x|good) / p(x|bad) ratio
- More efficient than random/grid search: fewer trials needed

**Etymology**: "Parzen" from Emanuel Parzen (1929-), who developed kernel density estimation

### Why GroupKFold?

**Problem**: Labels are **user-level**, but data is **tweet-level**
- Same user's tweets share the same label
- Random splitting would cause **data leakage**: model sees same user in both train/val

**Solution**: GroupKFold ensures all tweets from same user stay in same fold
- Prevents leakage: model never sees same user in both sets
- More realistic validation: simulates real-world scenario

**Data Leakage**: 
When information from test/validation set leaks into training, causing overly optimistic performance estimates.

### Why OrdinalEncoder vs One-Hot?

**Tree Model Efficiency**:
- **One-Hot Encoding**: Creates sparse binary features (many zeros)
  - Tree models can handle this, but less efficient
  - Increases dimensionality significantly
  
- **Ordinal Encoding**: Maps categories to integers (0, 1, 2, ...)
  - Tree models handle ordinal splits naturally
  - More compact representation
  - Better for high-cardinality categorical features (like `source`)

**Rule of Thumb**: Use OrdinalEncoder for tree models, One-Hot for linear models

## Feature Engineering

### Phase 1: Network Asymmetry Features

**Core Feature**: `follower_following_ratio`
- Formula: `user.followers_count / (user.friends_count + 1e-6)`
- **Influencer Signal**: High ratio indicates asymmetric network (many followers, few following)
- **Observer Signal**: Low ratio indicates reciprocal network (similar followers/following)

**Additional**: `follower_following_diff`
- Absolute difference: `followers_count - friends_count`

### Phase 2: Behavioral Rates

**Purpose**: Eliminate temporal collinearity (account age confounds activity metrics)

**Features**:
- `account_age_days`: `(current_time - user.created_at).days`
- `tweets_per_day`: `user.statuses_count / (account_age_days + 1e-6)`
- `listed_per_day`: `user.listed_count / (account_age_days + 1e-6)`

**Why Rates?**: Normalizes activity by account age, revealing true behavioral patterns

### Phase 3: Tweet-level Features

- `is_reply`: 1 if `in_reply_to_user_id` not null (Observer signal - reciprocal conversation)
- `mentions_count`, `hashtags_count`, `urls_count`: From `entities` field lengths

### Phase 4: Categorical Encoding

- `source`: OrdinalEncoder (NOT One-Hot) for tree model efficiency

## Installation

```bash
cd src/xgboost/solution/
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
cd src/xgboost/solution/
python main.py \
    --train-file ../../../data/train.jsonl \
    --test-file ../../../data/kaggle_test.jsonl \
    --output ../../../output/submission.csv
```

### With Hyperparameter Optimization

```bash
python main.py \
    --train-file ../../../data/train.jsonl \
    --test-file ../../../data/kaggle_test.jsonl \
    --output ../../../output/submission.csv \
    --optimize \
    --n-trials 100 \
    --use-gpu
```

### Options

- `--optimize`: Enable Optuna hyperparameter optimization
- `--n-trials`: Number of optimization trials (default: 100)
- `--timeout`: Optimization timeout in seconds
- `--use-gpu`: Use GPU acceleration (default: True)
- `--no-gpu`: Disable GPU acceleration
- `--cv-folds`: Number of GroupKFold folds (default: 5)
- `--shap`: Run SHAP analysis (default: True)
- `--shap-samples`: Maximum samples for SHAP (default: 1000)

## GPU Configuration

The solution automatically detects GPU availability:
- Checks CUDA via PyTorch
- Tests XGBoost GPU support
- Falls back to CPU if GPU unavailable

**XGBoost 2.0+ API**:
- `tree_method='hist'` with `device='cuda'`

**Legacy API** (if needed):
- `tree_method='gpu_hist'`

## Output Files

1. **Submission CSV**: `output/submission.csv`
   - Format: `ID, Prediction`
   
2. **SHAP Analysis** (if enabled):
   - `output/shap/summary_plot.png`: Beeswarm plot
   - `output/shap/bar_plot.png`: Feature importance bar chart
   - `output/shap/feature_importance.csv`: Feature rankings

## Mathematical Formulations

### Feature Engineering Formulas

**Network Asymmetry**:
```
follower_following_ratio = followers_count / (friends_count + ε)
follower_following_diff = followers_count - friends_count
```

**Behavioral Rates**:
```
account_age_days = (current_time - user.created_at).days
tweets_per_day = statuses_count / (account_age_days + ε)
listed_per_day = listed_count / (account_age_days + ε)
```

**Where ε = 1e-6** (epsilon to prevent division by zero)

### XGBoost Loss Function

**Binary Logistic Loss**:
```
L(y, ŷ) = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]
```

**With Regularization**:
```
Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
```

Where Ω(f_k) = γ·T + (1/2)·λ·||w||² (L2 regularization)

### Bayesian Optimization

**Posterior Update**:
```
p(f(x) | D_n) ∝ p(D_n | f(x)) · p(f(x))
```

**Acquisition Function** (Expected Improvement):
```
EI(x) = E[max(0, f(x) - f(x*))]
```

## File Structure

```
solution/
├── utils.py                      # GPU detection, data loading utilities
├── feature_engineering.py       # Metadata feature extraction
├── validation.py                 # GroupKFold cross-validation
├── hyperparameter_optimization.py # Optuna Bayesian optimization
├── shap_analysis.py              # SHAP feature interpretability
├── main.py                       # Complete pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Principles

1. **Zero Dependencies**: Independent of existing `src/` code
2. **GPU-First**: All models use GPU acceleration when available
3. **Production-Ready**: Comprehensive error handling and logging
4. **Educational**: Theoretical explanations
5. **Blueprint Compliance**: Follows technical specifications exactly

## Troubleshooting

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Test XGBoost GPU: See `utils.detect_gpu_for_xgboost()`

### Out of Memory
- Reduce `--shap-samples` for SHAP analysis
- Use CPU mode: `--no-gpu`
- Reduce `--n-trials` for optimization

### Data Leakage Warning
- Ensure user IDs are correctly extracted
- Verify GroupKFold is working correctly
- Check that same users don't appear in both train/val sets

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
2. Bergstra, J., et al. (2011). Algorithms for Hyper-Parameter Optimization. NIPS.
3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS.

## License

This solution is provided for educational purposes as part of the Kaggle competition.

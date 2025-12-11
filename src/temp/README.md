# Distribution Shift Detection Module
# 

This module detects distribution shift between training and test sets by training a classifier to distinguish between them.

## Overview

If a classifier can easily distinguish between training and test sets (high AUC), it indicates that feature engineering may have introduced distribution shift, which can harm model generalization.

## Usage

### Basic Usage

```bash
python src/temp/main.py
```

### With Custom Parameters

```bash
python src/temp/main.py \
    --n-splits 5 \
    --balance-samples \
    --save-model \
    --top-n 50
```

### Command Line Arguments

- `--train-features`: Path to train_features.pkl (default: output/preprocessing/train_features.pkl)
- `--test-features`: Path to test_features.pkl (default: output/preprocessing/test_features.pkl)
- `--balance-samples`: Balance the number of samples from train and test sets
- `--n-splits`: Number of folds for cross-validation (default: 5)
- `--random-state`: Random seed (default: 42)
- `--max-depth`: XGBoost max_depth parameter (default: 6)
- `--learning-rate`: XGBoost learning_rate parameter (default: 0.1)
- `--n-estimators`: XGBoost n_estimators parameter (default: 100)
- `--save-model`: Save the trained model
- `--output-dir`: Output directory (default: output/temp)
- `--top-n`: Number of top features to display (default: 50)

## Output Files

Results are saved to `output/temp/`:

- `distribution_shift_classifier.pkl`: Trained model (if --save-model is used)
- `feature_importance.csv`: Feature importance sorted by importance
- `summary.json`: Summary of metrics and interpretation

## Interpretation

- **AUC > 0.9**: HIGH distribution shift - Feature engineering may have introduced significant bias
- **AUC > 0.7**: MODERATE distribution shift - Some features may be problematic
- **AUC < 0.7**: LOW distribution shift - Features appear consistent

## Module Structure

- `load_data.py`: Loads training and test features, creates labels, and merges them
- `train_classifier.py`: Trains XGBoost classifier and evaluates performance
- `analyze_importance.py`: Analyzes and outputs feature importance
- `main.py`: Main entry point with command-line interface


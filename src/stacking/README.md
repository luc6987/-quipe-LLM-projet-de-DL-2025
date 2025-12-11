# Stacking Ensemble Module

##  / Overview

boostingLightGBM + XGBoost + CatBoost

This module implements stacking ensemble for three boosting models: LightGBM + XGBoost + CatBoost.

##  / Usage

### 1: main.pysetup.yml

1.  `setup.yml` stacking
```yaml
stacking:
  enabled: true  # truestacking
  n_folds: 5
  text_components: 128
```

2. 
```bash
python src/main.py
```

### 2: stacking

```bash
python src/stacking/main.py
```



```bash
python -m src.stacking.main
```

##  / Configuration

 `setup.yml`  `stacking` 

- `enabled`: stacking: false
- `n_folds`: : 5
- `text_components`: PCA: 128

##  / Output Files

 `output/stacking/` 

- `correlations_oof.csv`: OOF
- `correlations_test.csv`: 
- `oof_predictions.npz`: OOF logits
- `test_predictions.npz`: logits
- `meta_model_info.json`: 

 `output/submissions/submission_stacking_boosting.csv`

##  / Workflow

1. **OOF**: LightGBM, XGBoost, CatBoost5OOF logits
2. ****: Pearson
3. ****: OOF logitsLogistic Regression
4. ****: 

##  / Notes

- Stacking
-  `setup.yml` 
- GPU


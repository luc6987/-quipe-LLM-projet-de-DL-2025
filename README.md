# Kaggle 2025 “Influencers or Observers” Guide

End-to-end pipeline for the Kaggle competition “Influencers or Observers: Predicting Social Roles”: unified preprocessing, three tree models (XGBoost/LightGBM/CatBoost), SHAP explainability, and submission generation (manual or auto).

## 1) What this project does (one-liner)
Takes the official JSONL train/test → shared feature engineering (text embeddings + metadata) → trains the selected model → produces `submission.csv`, with optional auto-submit to Kaggle.

## 2) Configure Kaggle API
1. In Kaggle account settings: `Account` → `Create New API Token`, get `kaggle.json`.  
2. Place the token and set permissions:  
macOS/Linux:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
Windows: `%USERPROFILE%\.kaggle\kaggle.json`  
3. Verify:
```bash
kaggle competitions list
```
If you see competition entries, the API is ready.

## 3) One-click env + data with `initialize.py`
Run from project root (recommended for first-time setup):
```bash
python initialize.py
```
It will:
- Create `.env` from `.kaggle/kaggle.json` (writing `KAGGLE_USERNAME/KEY` and competition name).
- Create a local env (conda preferred, otherwise venv) and install `requirements.txt`.
- Download competition data via Kaggle API and unpack to `data/train.jsonl` and `data/kaggle_test.jsonl`.
- Create needed folders (`output/`, caches, embeddings, visualizations, etc.).

If you already have a global environment, you may skip the install step, but keep `.env` and the downloaded data.

## 4) Configure `setup.yml` and run the main pipeline
- Key switches:
  - `model.active`: choose `xgboost`, `lightgbm`, or `catboost`.
  - `<model>.output_file`: submission filename, saved to `output/submissions/`.
  - `<model>.submit`: set to `true` to auto-submit after training (requires Kaggle API set up).
  - Data paths: `<model>.train_file` / `<model>.test_file` default to `data/`.
- Run (use the env created by the script):
```bash
source .env/bin/activate          # for venv; or: conda activate ./.env
python src/main.py --config setup.yml
```
- Or simply
```
.env/bin/python src/main.py
```
The script completes preprocessing, training, monitoring, and writes `output/submissions/<output_file>`. If `shap` is enabled, SHAP plots are saved to `output/visualization/shap/`.

## 5) Submit via Kaggle CLI
Manual submission (example `submission.csv`):
```bash
kaggle competitions submit \
  -c influencers-or-observers-predicting-social-roles \
  -f output/submissions/submission.csv \
  -m "my first submission"
```
If `<model>.submit: true` in `setup.yml`, the main script will run the same command automatically after training (API configured and competition rules accepted).

## Quick directory tour
- `initialize.py`: one-click env + data
- `setup.yml`: global config (paths, features, models, submit flags)
- `src/main.py`: main entry to run the full pipeline
- `output/`: monitoring, visualizations, submissions
- `data/`: `train.jsonl`, `kaggle_test.jsonl` (downloaded by script)

## Common issues
- Missing `kaggle.json`: ensure it is under `~/.kaggle/` with `chmod 600`.
- 401 when downloading: accept competition rules; refresh token if expired.
- Permission to run script: `chmod +x initialize.py` or use `python initialize.py`.
- GPU settings: toggle `use_gpu` in each model block; set to `false` if no GPU.  

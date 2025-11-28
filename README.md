# Kaggle 2025 Competition: Influencers or Observers

This project contains code and experiments for the Kaggle competition "Influencers or Observers: Predicting Social Roles".

## 📁 Project Structure

```
Kaggle2025/
├── README.md                    # Project documentation (this file)
├── initiate.py                  # Initialization script (download data, install dependencies)
├── requirements.txt             # Python dependency list
├── .gitignore                   # Git ignore file configuration
│
├── data/                        # Data directory (created automatically by initiate.py)
│   ├── train.jsonl             # Training data
│   └── kaggle_test.jsonl       # Test data
│
├── cache/                       # Cache directory (intermediate processing results)
│   ├── extracted_features.csv
│   ├── cleaned_features.csv
│   ├── test_extracted_features.csv
│   ├── test_cleaned_features.csv
│   ├── test_ids.csv
│   └── train_labels.csv
│
├── output/                      # Output directory (prediction results)
│   └── xgboost_baseline.csv    # XGBoost baseline model prediction results
│
└── src/                         # Source code directory
    └── xgboost/                 # XGBoost related code
        ├── xgboost_baseline.py  # XGBoost baseline model main file
        ├── train_xgboost.py    # Training script
        ├── test_xgboost.py     # Test script
        ├── extract_features.py # Feature extraction
        ├── clean.py            # Data cleaning
        └── README_XGBOOST.md   # XGBoost strategy documentation
```

## 🚀 Quick Start

### 1. Configure Kaggle API

Before using `initiate.py` to download data, you need to configure Kaggle API credentials first.

#### Step 1: Get API Token

1. Log in to [Kaggle](https://www.kaggle.com/)
2. Go to [Account Settings](https://www.kaggle.com/account)
3. In the "API" section, click "Create New Token"
4. This will download a `kaggle.json` file

#### Step 2: Place API Token

Place the `kaggle.json` file in the following location based on your operating system:

**macOS / Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions
```

**Windows:**
```powershell
# Create directory (if it doesn't exist)
mkdir %USERPROFILE%\.kaggle

# Move file (assuming download is in Downloads folder)
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

#### Step 3: Verify Configuration

Run the following command to verify the configuration is successful:

```bash
kaggle competitions list
```

If you can see the competition list, the configuration is successful!

### 2. Run Initialization Script

The `initiate.py` script will automatically complete the following tasks:

1. ✅ **Install Dependencies**: Install all required Python packages from `requirements.txt`
2. ✅ **Download Competition Data**: Download competition data files from Kaggle
3. ✅ **Extract Files**: Automatically extract downloaded zip files
4. ✅ **Organize File Structure**: Move `train.jsonl` and `kaggle_test.jsonl` to `data/` directory
5. ✅ **Clean Temporary Files**: Delete extra zip files and other temporary files

#### Usage

```bash
# Method 1: Run with Python
python initiate.py

# Method 2: If executable permissions are set (macOS/Linux)
./initiate.py

# Method 3: Explicitly specify Python 3
python3 initiate.py
```

#### Script Output Example

```
============================================================
Kaggle Competition Setup Script
============================================================

Step 1: Installing dependencies...
============================================================
Installing packages from requirements.txt...
============================================================
Found 5 packages to install:
  - numpy
  - pandas
  - xgboost
  - scikit-learn
  - kaggle

Running: python -m pip install numpy pandas xgboost scikit-learn kaggle
Installation completed successfully!

Step 2: Downloading Kaggle data...
============================================================
Downloading Kaggle competition data...
============================================================
Running: kaggle competitions download -c influencers-or-observers-predicting-social-roles
Download completed!

Step 3: Extracting and organizing files...
============================================================
Extracting and organizing files...
============================================================
Created/verified data directory: /path/to/Kaggle2025/data
Extracting influencers-or-observers-predicting-social-roles.zip...
  Extracted influencers-or-observers-predicting-social-roles.zip
  Moved kaggle_test.jsonl to data/
  Moved train.jsonl to data/

Cleaning up extra files...
  Deleted influencers-or-observers-predicting-social-roles.zip

============================================================
Verification:
============================================================
✓ kaggle_test.jsonl found in data/
✓ train.jsonl found in data/

============================================================
Setup completed successfully!
============================================================
```

### 3. Run Model

After initialization is complete, you can run model training and prediction:

```bash
# Enter source code directory
cd src/xgboost

# Run XGBoost baseline model
python xgboost_baseline.py
```

## 📦 Dependencies

The Python packages required for this project are listed in `requirements.txt`:

- `numpy` - Numerical computing
- `pandas` - Data processing
- `xgboost` - Gradient boosting decision tree model
- `scikit-learn` - Machine learning tools
- `kaggle` - Kaggle API client

All dependencies will be automatically installed when running `initiate.py`.

## 🔧 Manual Installation (Optional)

If you don't want to use `initiate.py`, you can also install manually:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install numpy pandas xgboost scikit-learn kaggle
```

## ⚠️ Common Issues

### 1. Kaggle API Authentication Failed

**Error Message:**
```
Error: Could not find kaggle.json. Make sure it's located in ~/.kaggle/
```

**Solution:**
- Ensure `kaggle.json` file is in the correct location (see configuration steps above)
- Check file permissions (macOS/Linux need to be set to 600)
- Verify JSON file format is correct

### 2. Download Failed

**Error Message:**
```
Error downloading data: 401 - Unauthorized
```

**Solution:**
- Check if Kaggle API token has expired (need to re-download)
- Confirm Kaggle account has accepted competition rules
- Verify network connection

### 3. Permission Error (macOS/Linux)

**Error Message:**
```
Permission denied: ./initiate.py
```

**Solution:**
```bash
chmod +x initiate.py
```

### 4. Python Version Issue

**Requirement:** Python 3.7 or higher

Check Python version:
```bash
python --version
# or
python3 --version
```

## 📝 Notes

1. **Data Files**: Files in the `data/` directory are large and have been added to `.gitignore`, so they won't be tracked by Git
2. **Cache Files**: The `cache/` directory contains intermediate processing results and can be deleted and regenerated
3. **Output Files**: The `output/` directory contains model prediction results for submission to Kaggle

## 🔗 Related Links

- [Kaggle Competition Page](https://www.kaggle.com/competitions/influencers-or-observers-predicting-social-roles)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 License

This project is for learning and research purposes only.

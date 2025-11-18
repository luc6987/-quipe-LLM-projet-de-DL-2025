"""
XGBoost训练和预测脚本 / XGBoost Training and Prediction Script
完整流程：特征提取 -> 数据清洗 -> 模型训练 -> 预测 / Complete pipeline: Feature extraction -> Data cleaning -> Model training -> Prediction
"""

import os
import sys
# 添加 /Data 目录中的包路径（如果存在） / Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold  # 改为GroupKFold / Changed to GroupKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 tqdm 用于显示进度条 / Try to import tqdm for progress bar display
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna is not installed, cannot perform hyperparameter optimization. Use 'pip install optuna' to install.")

# 导入新模块 / Import new modules
try:
    from .baseline_model import create_baseline_xgboost_params, detect_gpu_for_xgboost
    from .validation import group_kfold_cv
    from .hyperparameter_optimization import optimize_hyperparameters
except ImportError:
    # 如果相对导入失败，尝试绝对导入 / If relative import fails, try absolute import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from baseline_model import create_baseline_xgboost_params, detect_gpu_for_xgboost
    from validation import group_kfold_cv
    from hyperparameter_optimization import optimize_hyperparameters

# 导入特征提取和清洗模块（将在main函数中动态导入） / Import feature extraction and cleaning modules (will be dynamically imported in main function)


def safe_read_csv(file_path, chunk_size=None, max_size_mb=1000):
    """
    安全地读取CSV文件，支持大文件和错误处理 / Safely read CSV file, supports large files and error handling
    
    Args:
        file_path: CSV文件路径 / CSV file path
        chunk_size: 分块读取大小（行数），None表示一次性读取 / Chunk size (number of rows), None means read all at once
        max_size_mb: 如果文件超过此大小（MB），使用分块读取 / If file exceeds this size (MB), use chunked reading
        
    Returns:
        DataFrame: 读取的数据 / Read data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # 检查文件大小 / Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Reading file: {file_path} (size: {file_size_mb:.2f} MB)")
    
    try:
        if chunk_size is None and file_size_mb > max_size_mb:
            print("  File is large, using chunked reading...")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ Reading complete: {df.shape}")
            return df
        elif chunk_size is not None:
            print(f"  Using chunked reading (chunk_size={chunk_size})...")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ Reading complete: {df.shape}")
            return df
        else:
            df = pd.read_csv(file_path)
            print(f"  ✓ Reading complete: {df.shape}")
            return df
    except Exception as e:
        print(f"  Error: Failed to read file: {str(e)}")
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            print("  Hint: File read timeout, trying chunked reading...")
            try:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                print(f"  ✓ Chunked reading successful: {df.shape}")
                return df
            except Exception as e2:
                raise IOError(f"Chunked reading also failed: {str(e2)}")
        else:
            raise


def load_labels_and_ids(train_labels_file='cache/train_labels.csv',
                       test_ids_file='cache/test_ids.csv'):
    """
    加载训练集标签和测试集ID / Load training set labels and test set IDs
    
    Args:
        train_labels_file: 训练集标签文件路径 / Training set label file path
        test_ids_file: 测试集ID文件路径 / Test set ID file path
        
    Returns:
        labels: 训练集标签Series / Training set label Series
        test_ids: 测试集ID Series / Test set ID Series
    """
    print("\n" + "="*60)
    print("Loading Labels and IDs")
    print("="*60)
    
    # 加载标签 / Load labels
    if os.path.exists(train_labels_file):
        try:
            labels_df = safe_read_csv(train_labels_file, max_size_mb=100)
            labels = labels_df['label']
            print(f"✓ Loaded training set labels: {len(labels)} records")
        except Exception as e:
            print(f"Error: Failed to read label file: {str(e)}")
            labels = None
    else:
        print(f"Error: Label file not found: {train_labels_file}")
        labels = None
    
    # 加载测试集ID / Load test set IDs
    if os.path.exists(test_ids_file):
        try:
            test_ids_df = safe_read_csv(test_ids_file, max_size_mb=100)
            test_ids = test_ids_df['challenge_id']
            print(f"✓ Loaded test set IDs: {len(test_ids)} records")
        except Exception as e:
            print(f"Error: Failed to read test set ID file: {str(e)}")
            test_ids = None
    else:
        print(f"Warning: Test set ID file not found: {test_ids_file}")
        test_ids = None
    
    return labels, test_ids


def train_xgboost_model(X_train, y_train, groups=None, cv_folds=5, xgb_params=None, use_gpu=True):
    """
    训练XGBoost模型并进行GroupKFold交叉验证 / Train XGBoost model and perform GroupKFold cross-validation
    
    关键修改：使用GroupKFold替代StratifiedKFold，防止数据泄露 / Key modification: Use GroupKFold instead of StratifiedKFold to prevent data leakage
    
    Args:
        X_train: 训练特征DataFrame / Training feature DataFrame
        y_train: 训练标签Series / Training label Series
        groups: 用户ID数组（用于GroupKFold，防止数据泄露） / User ID array (for GroupKFold, prevents data leakage)
        cv_folds: 交叉验证折数 / Number of cross-validation folds
        xgb_params: XGBoost参数字典，如果为None则使用默认参数 / XGBoost parameter dict, if None use default params
        use_gpu: 是否使用GPU加速 / Whether to use GPU acceleration
        
    Returns:
        model: 训练好的最终模型 / Trained final model
        cv_scores: 交叉验证分数列表 / Cross-validation score list
    """
    print("\n" + "="*60)
    print("Training XGBoost Model (GroupKFold Validation)")
    print("="*60)
    
    # 转换为numpy数组 / Convert to numpy arrays
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    # 如果没有提供groups，警告用户 / If groups not provided, warn user
    if groups is None:
        print("⚠ Warning: groups parameter not provided, cannot use GroupKFold")
        print("  This may cause data leakage! Recommend providing user IDs for GroupKFold")
        # 回退到StratifiedKFold（不推荐） / Fallback to StratifiedKFold (not recommended)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        use_group_kfold = False
    else:
        # 使用GroupKFold（推荐） / Use GroupKFold (recommended)
        gkf = GroupKFold(n_splits=cv_folds)
        use_group_kfold = True
        if isinstance(groups, pd.Series):
            groups_array = groups.values
        else:
            groups_array = groups
        print(f"  Using GroupKFold, unique users: {len(np.unique(groups_array))}")
    
    # 默认XGBoost参数（使用基线模型参数） / Default XGBoost parameters (use baseline model params)
    if xgb_params is None:
        xgb_params = create_baseline_xgboost_params(
            y_train=y_train_array,
            use_gpu=use_gpu
        )
    
    # 检测GPU并更新tree_method和device参数 / Detect GPU and update tree_method and device parameters
    if use_gpu:
        gpu_available, tree_method = detect_gpu_for_xgboost()
        xgb_params['tree_method'] = tree_method
        # XGBoost 2.0+ 版本：如果GPU可用且tree_method是'hist'，添加device='cuda' / XGBoost 2.0+: If GPU available and tree_method is 'hist', add device='cuda'
        if gpu_available and tree_method == 'hist' and 'device' not in xgb_params:
            xgb_params['device'] = 'cuda'
    
    print("\nXGBoost Parameters:")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    # 交叉验证 / Cross-validation
    if use_group_kfold:
        print(f"\nRunning {cv_folds}-Fold GroupKFold Cross-Validation...")
        cv_scores = []
        
        # 使用 tqdm 显示进度（如果可用） / Use tqdm to show progress (if available)
        fold_iterator = enumerate(gkf.split(X_train_array, y_train_array, groups=groups_array), 1)
        if TQDM_AVAILABLE:
            fold_iterator = tqdm(fold_iterator, total=cv_folds, desc="Cross-validation progress", unit="fold")
        
        for fold, (train_idx, val_idx) in fold_iterator:
            if not TQDM_AVAILABLE:
                print(f"\nFold {fold}/{cv_folds}:")
            else:
                print(f"\nTraining Fold {fold}/{cv_folds}...")
            
            X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
            y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
            
            # 创建XGBoost模型 / Create XGBoost model
            model = xgb.XGBClassifier(**xgb_params)
            
            # 训练（verbose=10 表示每10轮显示一次训练进度） / Training (verbose=10 means show progress every 10 rounds)
            n_estimators = xgb_params.get('n_estimators', 100)
            verbose_interval = max(1, n_estimators // 20)  # 大约显示20次进度更新 / Approximately show 20 progress updates
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=verbose_interval
            )
            
            # 预测和评估 / Prediction and evaluation
            y_pred = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(acc)
            
            print(f"  ✓ Fold {fold} Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    else:
        print(f"\nRunning {cv_folds}-Fold StratifiedKFold Cross-Validation (not recommended, may cause data leakage)...")
        cv_scores = []
        
        # 使用 tqdm 显示进度（如果可用） / Use tqdm to show progress (if available)
        fold_iterator = enumerate(skf.split(X_train_array, y_train_array), 1)
        if TQDM_AVAILABLE:
            fold_iterator = tqdm(fold_iterator, total=cv_folds, desc="Cross-validation progress", unit="fold")
        
        for fold, (train_idx, val_idx) in fold_iterator:
            if not TQDM_AVAILABLE:
                print(f"\nFold {fold}/{cv_folds}:")
            else:
                print(f"\nTraining Fold {fold}/{cv_folds}...")
            
            X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
            y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
            
            # 创建XGBoost模型 / Create XGBoost model
            model = xgb.XGBClassifier(**xgb_params)
            
            # 训练（verbose=10 表示每10轮显示一次训练进度） / Training (verbose=10 means show progress every 10 rounds)
            n_estimators = xgb_params.get('n_estimators', 100)
            verbose_interval = max(1, n_estimators // 20)  # 大约显示20次进度更新 / Approximately show 20 progress updates
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=verbose_interval
            )
            
            # 预测和评估 / Prediction and evaluation
            y_pred = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(acc)
            
            print(f"  ✓ Fold {fold} Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores)*100:.2f}%)")
    print(f"  Std Deviation: {np.std(cv_scores):.4f} ({np.std(cv_scores)*100:.2f}%)")
    print(f"  Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"{'='*60}")
    
    # 训练最终模型（使用全部训练数据） / Train final model (using all training data)
    print("\nTraining final model on all training data...")
    final_model = xgb.XGBClassifier(**xgb_params)
    # 启用训练进度显示 / Enable training progress display
    n_estimators = xgb_params.get('n_estimators', 100)
    verbose_interval = max(1, n_estimators // 20)  # 大约显示20次进度更新 / Approximately show 20 progress updates
    final_model.fit(X_train_array, y_train_array, verbose=verbose_interval)
    print("✓ Final model training complete")
    
    return final_model, cv_scores


def optimize_hyperparameters_wrapper(X_train, y_train, groups=None, cv_folds=5, 
                                    n_trials=50, timeout=None, use_gpu=True):
    """
    使用Optuna进行XGBoost超参数优化（包装函数，使用GroupKFold）
    
    Args:
        X_train: 训练特征DataFrame
        y_train: 训练标签Series
        groups: 用户ID数组（用于GroupKFold）
        cv_folds: 交叉验证折数
        n_trials: Optuna优化试验次数
        timeout: 优化超时时间（秒），None表示不限制
        use_gpu: 是否使用GPU
        
    Returns:
        best_params: 最佳参数字典
        best_score: 最佳交叉验证分数
    """
    # 使用hyperparameter_optimization模块中的函数
    return optimize_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        n_trials=n_trials,
        cv_folds=cv_folds,
        use_gpu=use_gpu,
        timeout=timeout,
        verbose=True
    )
    
    # 旧的optimize_hyperparameters函数已删除，使用hyperparameter_optimization模块中的函数


def make_predictions(model, X_test, test_ids, output_file='output/xgboost_baseline.csv'):
    """
    生成预测结果并保存为提交文件 / Generate predictions and save as submission file
    
    Args:
        model: 训练好的模型 / Trained model
        X_test: 测试特征DataFrame / Test feature DataFrame
        test_ids: 测试集ID Series / Test set ID Series
        output_file: 输出文件路径 / Output file path
        
    Returns:
        submission: 提交DataFrame / Submission DataFrame
    """
    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)
    
    # 确保所有列都是数值类型（处理可能遗漏的字符串列） / Ensure all columns are numeric (handle possibly missed string columns)
    non_numeric_cols = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric_cols) > 0:
        print(f"Warning: Found {len(non_numeric_cols)} non-numeric columns, converting to numeric type...")
        for col in non_numeric_cols:
            # 使用pd.Categorical进行编码，将字符串转换为数值 / Use pd.Categorical to encode, convert strings to numeric
            X_test[col] = pd.Categorical(X_test[col]).codes
            print(f"  Encoded column: {col}")
    
    X_test_array = X_test.values
    predictions = model.predict(X_test_array)
    
    # 创建提交文件（格式：ID, Prediction） / Create submission file (format: ID, Prediction)
    submission = pd.DataFrame({
        'ID': test_ids,
        'Prediction': predictions
    })
    
    # 确保输出目录存在 / Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 保存提交文件 / Save submission file
    submission.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to: {output_file}")
    print(f"  Number of prediction samples: {len(predictions)}")
    print("  Prediction value distribution:")
    print(submission['Prediction'].value_counts().sort_index())
    
    return submission


def main():
    """
    主函数：执行完整的训练和预测流程 / Main function: Execute complete training and prediction pipeline
    """
    import argparse
    
    # 获取项目根目录（从 src/xgboost 向上两级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 切换到项目根目录
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    # 将src/xgboost目录添加到Python路径，以便导入模块
    xgboost_dir = os.path.join(project_root, 'src', 'xgboost')
    if xgboost_dir not in sys.path:
        sys.path.insert(0, xgboost_dir)
    
    # 导入特征提取和清洗模块
    from extract_features import extract_all_features, extract_labels_and_ids
    from clean import clean_and_reduce
    
    try:
        parser = argparse.ArgumentParser(description='XGBoost training and prediction script')
        parser.add_argument('--train-file', type=str, default='data/train.jsonl',
                            help='Training set JSONL file path')
        parser.add_argument('--test-file', type=str, default='data/kaggle_test.jsonl',
                            help='Test set JSONL file path')
        parser.add_argument('--run-extract', action='store_true', default=False,
                            help='Run feature extraction step (default: skip, use existing feature files)')
        parser.add_argument('--run-clean', action='store_true', default=False,
                            help='Run data cleaning step (default: skip, use existing cleaned files)')
        parser.add_argument('--cv-folds', type=int, default=5,
                            help='Number of cross-validation folds')
        parser.add_argument('--output', type=str, default='output/xgboost_baseline.csv',
                            help='Output submission file path')
        parser.add_argument('--optimize', action='store_true', default=False,
                            help='Use Optuna for hyperparameter optimization (recommended for performance improvement)')
        parser.add_argument('--n-trials', type=int, default=100,
                            help='Number of Optuna optimization trials (only effective with --optimize, default: 100)')
        parser.add_argument('--timeout', type=int, default=None,
                            help='Optuna optimization timeout in seconds, None means no limit (only effective with --optimize)')
        parser.add_argument('--quick-optimize', action='store_true', default=False,
                            help='Quick optimization mode: Use Optuna but only run 20 trials (for quick testing)')
        
        args = parser.parse_args()
        
        print("="*60)
        print("XGBoost Complete Training and Prediction Pipeline")
        print("="*60)
        print(f"Project root directory: {project_root}")
        
        # ========== Step 1: Feature Extraction ========== / 步骤1: 特征提取
        if args.run_extract:
            print("\n[Step 1] Feature Extraction")
            print("="*60)
            
            # 提取训练集特征和标签 / Extract training set features and labels
            train_features_df, train_feature_cols, _, user_ids, transformers = extract_all_features(
                args.train_file,
                'cache/extracted_features.csv',
                exclude_label=True,
                exclude_id=True
            )
            
            # 保存用户ID（用于GroupKFold） / Save user IDs (for GroupKFold)
            if user_ids is not None:
                user_ids_df = pd.DataFrame({'user_id': user_ids})
                user_ids_df.to_csv('cache/train_user_ids.csv', index=False)
                print("✓ User IDs saved to: cache/train_user_ids.csv")
            
            # 提取训练集标签 / Extract training set labels
            labels, _ = extract_labels_and_ids(
                args.train_file,
                output_labels_file='cache/train_labels.csv'
            )
            
            # 提取测试集特征和ID / Extract test set features and IDs
            # 注意：测试集需要使用训练集的transformers（特别是source_encoder） / Note: Test set needs to use training set transformers (especially source_encoder)
            test_features_df, test_feature_cols, test_challenge_ids, _, _ = extract_all_features(
                args.test_file,
                'cache/test_extracted_features.csv',
                exclude_label=True,
                exclude_id=False
            )
            
            # 保存测试集ID（extract_all_features已经分离了challenge_id） / Save test set IDs (extract_all_features already separated challenge_id)
            if test_challenge_ids is not None:
                ids_df = pd.DataFrame({'challenge_id': test_challenge_ids})
                ids_df.to_csv('cache/test_ids.csv', index=False)
                print("✓ Test set IDs saved to: cache/test_ids.csv")
        else:
            print("\n[Step 1] Skipping feature extraction (using existing files)")
            labels = None
            test_challenge_ids = None
        
        # ========== Step 2: Data Cleaning ========== / 步骤2: 数据清洗
        if args.run_clean:
            print("\n[Step 2] Data Cleaning and Dimensionality Reduction")
            print("="*60)
            
            df_train_cleaned, transformers, df_test_cleaned = clean_and_reduce(
                input_file='cache/extracted_features.csv',
                output_file='cache/cleaned_features.csv',
                missing_strategy='nan',  # 改为'nan'，保留np.nan让XGBoost处理 / Changed to 'nan', keep np.nan for XGBoost to handle
                outlier_method='clip',
                remove_low_var=False,  # 禁用低方差特征移除，对XGBoost树模型通常不需要 / Disable low variance feature removal, usually not needed for XGBoost tree models
                remove_high_corr=False,  # 禁用高相关特征移除，XGBoost可以处理相关特征 / Disable high correlation feature removal, XGBoost can handle correlated features
                normalize_method='none',  # 禁用标准化，XGBoost树模型不需要标准化 / Disable normalization, XGBoost tree models don't need normalization
                reduce_method='none',  # 禁用降维，保留所有特征 / Disable dimensionality reduction, keep all features
                variance_ratio=0.95,
                test_input_file='cache/test_extracted_features.csv',
                test_output_file='cache/test_cleaned_features.csv'
            )
        else:
            print("\n[Step 2] Skipping data cleaning (using existing files)")
        
        # ========== Step 3: Load Data ========== / 步骤3: 加载数据
        print("\n[Step 3] Loading Cleaned Data and Labels")
        print("="*60)
        
        # 检查必需的文件是否存在 / Check if required files exist
        train_cleaned_file = 'cache/cleaned_features.csv'
        test_cleaned_file = 'cache/test_cleaned_features.csv'
        
        if not os.path.exists(train_cleaned_file):
            print(f"Error: Training set cleaned file not found: {train_cleaned_file}")
            print("Please run data cleaning step first (use --run-clean)")
            return
        
        if not os.path.exists(test_cleaned_file):
            print(f"Warning: Test set cleaned file not found: {test_cleaned_file}")
            print("Automatically running data cleaning step to generate test set cleaned file...")
            
            # 检查测试集提取文件是否存在 / Check if test set extraction file exists
            test_extracted_file = 'cache/test_extracted_features.csv'
            if not os.path.exists(test_extracted_file):
                print(f"Error: Test set extraction file not found: {test_extracted_file}")
                print("Please run feature extraction step first (use --run-extract)")
                return
            
            # 运行清洗步骤（仅处理测试集，使用已有的训练集清洗文件） / Run cleaning step (only process test set, use existing training set cleaned file)
            print("Loading transformer information from training set cleaned file...")
            # 由于我们需要完整的transformer信息，需要重新运行清洗 / Since we need complete transformer information, need to re-run cleaning
            # 但可以只读取训练集已清洗的数据，然后只处理测试集 / But can just read training set cleaned data, then only process test set
            try:
                df_train_cleaned = safe_read_csv(train_cleaned_file)
                print(f"Loaded training set cleaned data: {df_train_cleaned.shape}")
            except Exception as e:
                print(f"Error: Failed to read training set cleaned file: {str(e)}")
                return
            
            print("Warning: Need to re-run complete data cleaning pipeline to properly handle test set")
            print("Please run: python train_xgboost.py --run-clean")
            return
        
        # 加载清洗后的特征 / Load cleaned features
        print("\nLoading training set features...")
        X_train = safe_read_csv(train_cleaned_file)
        print("\nLoading test set features...")
        X_test = safe_read_csv(test_cleaned_file)
        
        print(f"Training set feature shape: {X_train.shape}")
        print(f"Test set feature shape: {X_test.shape}")
        
        # 加载标签和ID / Load labels and IDs
        if labels is None:
            labels, test_ids = load_labels_and_ids()
        else:
            test_ids = None
            if test_challenge_ids is not None:
                test_ids = pd.Series(test_challenge_ids)
            elif os.path.exists('cache/test_ids.csv'):
                try:
                    test_ids_df = safe_read_csv('cache/test_ids.csv', max_size_mb=100)
                    test_ids = test_ids_df['challenge_id']
                except Exception as e:
                    print(f"Error: Failed to read test set ID file: {str(e)}")
                    test_ids = None
        
        if labels is None:
            print("Error: Unable to load training set labels")
            return
        
        if test_ids is None:
            print("Error: Unable to load test set IDs")
            return
        
        # 确保标签和特征数量匹配 / Ensure label and feature counts match
        if len(labels) != len(X_train):
            print(f"Warning: Label count ({len(labels)}) does not match feature count ({len(X_train)})")
            # 取较小值 / Take the smaller value
            min_len = min(len(labels), len(X_train))
            labels = labels[:min_len]
            X_train = X_train.iloc[:min_len]
            print(f"Adjusted to: {min_len} records")
        
        # 记录最终的训练数据长度（用于后续对齐groups） / Record final training data length (for subsequent group alignment)
        final_train_len = len(X_train)
        
        # 确保测试集ID和特征数量匹配 / Ensure test set ID and feature counts match
        # 注意：测试集不应该修改行数，所以应该以ID数量为准 / Note: Test set should not modify row count, so should use ID count as reference
        if len(test_ids) != len(X_test):
            print(f"Warning: Test set ID count ({len(test_ids)}) does not match feature count ({len(X_test)})")
            if len(test_ids) > len(X_test):
                # 如果ID数量更多，说明特征数据被截断了，需要补充 / If ID count is more, feature data was truncated, need to supplement
                print(f"  Feature data has insufficient rows, will fill missing rows with last row")
                missing_rows = len(test_ids) - len(X_test)
                # 用最后一行填充 / Fill with last row
                last_row = X_test.iloc[-1:].copy()
                for _ in range(missing_rows):
                    X_test = pd.concat([X_test, last_row], ignore_index=True)
                print(f"  Added {missing_rows} rows, feature data now: {len(X_test)} rows")
            else:
                # 如果ID数量更少，截断特征数据以匹配ID / If ID count is less, truncate feature data to match IDs
                print(f"  Feature data has too many rows, will truncate to match ID count")
                X_test = X_test.iloc[:len(test_ids)]
                print(f"  Truncated to {len(test_ids)} rows")
            
            # 确保test_ids是Series以便正确使用 / Ensure test_ids is Series for proper use
            if not isinstance(test_ids, pd.Series):
                test_ids = pd.Series(test_ids)
        
        # 确保测试集特征列与训练集一致 / Ensure test set feature columns match training set
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        missing_cols = train_cols - test_cols
        extra_cols = test_cols - train_cols
        
        if missing_cols:
            print(f"  Test set missing {len(missing_cols)} columns, filling with 0")
            for col in missing_cols:
                X_test[col] = 0
        
        if extra_cols:
            print(f"  Test set has {len(extra_cols)} extra columns, removing")
            X_test = X_test.drop(columns=list(extra_cols))
        
        # 确保列顺序一致 / Ensure column order is consistent
        X_test = X_test[X_train.columns]
        
        print("✓ Data loading complete")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        
        # ========== Step 4: Model Training ========== / 步骤4: 模型训练
        print("\n[Step 4] Model Training")
        
        # 超参数优化 / Hyperparameter optimization
        xgb_params = None
        use_optimize = args.optimize or args.quick_optimize
        n_trials_to_use = 20 if args.quick_optimize else args.n_trials
        
        if use_optimize:
            if not OPTUNA_AVAILABLE:
                print("Error: optuna is not installed, cannot perform hyperparameter optimization.")
                print("Please use 'pip install optuna' to install, or remove --optimize parameter.")
                print("Will continue training with default parameters...")
            else:
                print(f"\n{'='*60}")
                print(f"Starting Hyperparameter Optimization (Number of trials: {n_trials_to_use})")
                print(f"{'='*60}")
                # 加载用户ID（用于GroupKFold） / Load user IDs (for GroupKFold)
                groups = None
                if os.path.exists('cache/train_user_ids.csv'):
                    groups = pd.read_csv('cache/train_user_ids.csv')['user_id'].values
                    # 确保groups长度与训练数据匹配 / Ensure groups length matches training data
                    if len(groups) != final_train_len:
                        print(f"  Warning: User ID count ({len(groups)}) does not match training data ({final_train_len}), will truncate to match length")
                        groups = groups[:final_train_len]
                    print(f"  Loaded user IDs: {len(np.unique(groups))} unique users")
                
                best_result = optimize_hyperparameters_wrapper(
                    X_train, labels, groups=groups,
                    cv_folds=args.cv_folds,
                    n_trials=n_trials_to_use,
                    timeout=args.timeout,
                    use_gpu=True
                )
                xgb_params = best_result['best_params']
                print("\nTraining final model with optimized parameters...")
        else:
            print("\nTip: You can use --optimize or --quick-optimize for hyperparameter optimization to improve performance")
            print("  --optimize: Full optimization (default 100 trials, slower but better results)")
            print("  --quick-optimize: Quick optimization (20 trials, faster but average results)")
        
        # 加载用户ID（用于GroupKFold） / Load user IDs (for GroupKFold)
        groups = None
        if os.path.exists('cache/train_user_ids.csv'):
            groups = pd.read_csv('cache/train_user_ids.csv')['user_id'].values
            # 确保groups长度与训练数据匹配 / Ensure groups length matches training data
            if len(groups) != final_train_len:
                print(f"  Warning: User ID count ({len(groups)}) does not match training data ({final_train_len}), will truncate to match length")
                groups = groups[:final_train_len]
            print(f"  Loaded user IDs: {len(np.unique(groups))} unique users")
        
        model, cv_scores = train_xgboost_model(
            X_train, labels, groups=groups, 
            cv_folds=args.cv_folds, 
            xgb_params=xgb_params,
            use_gpu=True
        )
        
        # ========== Step 5: Generate Predictions ========== / 步骤5: 生成预测
        print("\n[Step 5] Generate Predictions")
        submission = make_predictions(model, X_test, test_ids, output_file=args.output)
        
        # ========== Step 6: Feature Importance ========== / 步骤6: 特征重要性
        print("\n" + "="*60)
        print("Top 20 Most Important Features")
        print("="*60)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print(f"Submission file: {args.output}")
        print(f"Cross-validation mean accuracy: {np.mean(cv_scores):.4f}")
    finally:
        # 恢复原始工作目录 / Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()


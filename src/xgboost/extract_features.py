"""
提取train.jsonl和test.jsonl数据中所有194个特征 / Extract all 194 features from train.jsonl and test.jsonl data
将嵌套JSON展平为表格格式，排除文本特征 / Flatten nested JSON to tabular format, exclude text features
确保训练集和测试集有统一的特征列 / Ensure training and test sets have unified feature columns
"""

import json
import argparse
import os
from pandas import json_normalize
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入特征工程模块 / Import feature engineering module
try:
    from .feature_engineering import engineer_features, extract_user_id
except ImportError:
    # 如果相对导入失败，尝试绝对导入 / If relative import fails, try absolute import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import engineer_features, extract_user_id


def ensure_dir_exists(file_path):
    """
    确保文件路径的目录存在，如果不存在则创建 / Ensure directory of file path exists, create if not
    
    Args:
        file_path: 文件路径 / File path
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_all_feature_columns(train_file, test_file):
    """
    扫描训练集和测试集，获取所有可能的特征列 / Scan training and test sets to get all possible feature columns
    
    Args:
        train_file: 训练集JSONL文件路径 / Training set JSONL file path
        test_file: 测试集JSONL文件路径 / Test set JSONL file path
        
    Returns:
        list: 所有特征列的列表（排序后） / List of all feature columns (sorted)
    """
    print("Scanning training and test sets to determine all feature columns...")
    
    all_columns = set()
    
    # 需要排除的文本列 / Text columns to exclude
    exclude_cols = {
        'text',
        'extended_tweet.full_text',
        'quoted_status.extended_tweet.full_text',
        'quoted_status.text',
        'label',  # 训练集的标签列 / Training set label column
        'challenge_id'  # ID列（需要单独处理） / ID column (needs separate handling)
    }
    
    # 扫描训练集 / Scan training set
    if train_file:
        try:
            print(f"  Scanning training set: {train_file}")
            with open(train_file, 'r', encoding='utf-8') as f:
                sample_records = []
                for i, line in enumerate(f):
                    if i < 100:  # 只读取前100行来获取列结构
                        try:
                            data = json.loads(line.strip())
                            sample_records.append(data)
                        except json.JSONDecodeError:
                            continue
                    else:
                        break
            
            if sample_records:
                df_sample = json_normalize(sample_records)
                train_cols = set(df_sample.columns) - exclude_cols
                all_columns.update(train_cols)
                print(f"    Training set features: {len(train_cols)}")
        except Exception as e:
            print(f"    Warning: Error scanning training set: {e}")
    
    # 扫描测试集 / Scan test set
    if test_file:
        try:
            print(f"  Scanning test set: {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                sample_records = []
                for i, line in enumerate(f):
                    if i < 100:  # 只读取前100行来获取列结构
                        try:
                            data = json.loads(line.strip())
                            sample_records.append(data)
                        except json.JSONDecodeError:
                            continue
                    else:
                        break
            
            if sample_records:
                df_sample = json_normalize(sample_records)
                test_cols = set(df_sample.columns) - exclude_cols
                all_columns.update(test_cols)
                print(f"    Test set features: {len(test_cols)}")
        except Exception as e:
            print(f"    Warning: Error scanning test set: {e}")
    
    print(f"  Total features (after deduplication): {len(all_columns)}")
    return sorted(list(all_columns))  # 返回排序后的列名列表，确保顺序一致


def extract_all_features(input_file, output_file='extracted_features.csv',
                         exclude_label=True, exclude_id=True,
                         unified_feature_columns=None):
    """
    从JSONL文件中提取所有表格特征（排除文本列） / Extract all tabular features from JSONL file (exclude text columns)
    使用统一的特征列列表，确保训练集和测试集特征一致 / Use unified feature column list to ensure training and test sets have consistent features
    
    Args:
        input_file: 输入的JSONL文件路径 / Input JSONL file path
        output_file: 输出的CSV文件路径 / Output CSV file path
        exclude_label: 是否排除label列（默认True） / Whether to exclude label column (default True)
        exclude_id: 是否排除challenge_id列（默认True） / Whether to exclude challenge_id column (default True)
        unified_feature_columns: 统一的特征列列表（如果提供，确保输出包含所有列） / Unified feature column list (if provided, ensure output contains all columns)
    """
    print(f"Reading file: {input_file}")
    print("Flattening JSON data...")
    
    # 逐行读取并展平，更节省内存 / Read and flatten row by row, more memory efficient
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} rows...")
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON parsing failed for row {i+1}: {e}")
                continue
    
    print(f"  Successfully loaded {len(records)} records")
    print("  Normalizing JSON structure...")
    
    # 展平嵌套JSON / Flatten nested JSON
    df = json_normalize(records)
    
    print(f"  Original column count: {len(df.columns)}")
    print(f"  Data shape: {df.shape}")
    
    # 需要排除的文本列
    exclude_cols = [
        'text',
        'extended_tweet.full_text',
        'quoted_status.extended_tweet.full_text',
        'quoted_status.text'
    ]
    
    # 根据参数决定是否排除label和challenge_id
    if exclude_label and 'label' in df.columns:
        exclude_cols.append('label')
    if exclude_id and 'challenge_id' in df.columns:
        exclude_cols.append('challenge_id')
    
    # 获取当前文件中的特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 如果提供了统一的特征列列表，使用它来确保列的一致性
    if unified_feature_columns is not None:
        print(f"  Using unified feature column list (total {len(unified_feature_columns)} columns)")
        
        # 创建包含所有统一特征列的DataFrame / Create DataFrame containing all unified feature columns
        features_df = pd.DataFrame(index=df.index)
        
        # 填充存在的列 / Fill existing columns
        for col in unified_feature_columns:
            if col in df.columns:
                features_df[col] = df[col]
            else:
                # 如果列不存在，用NaN填充 / If column doesn't exist, fill with NaN
                features_df[col] = None
        
        # 确保列顺序与统一列表一致 / Ensure column order matches unified list
        features_df = features_df[unified_feature_columns]
        
        # 统计信息 / Statistics
        existing_cols = set(feature_cols) & set(unified_feature_columns)
        missing_cols = set(unified_feature_columns) - set(feature_cols)
        print(f"    Existing features: {len(existing_cols)}")
        if len(missing_cols) > 0:
            print(f"    Missing features (filled with NaN): {len(missing_cols)}")
    else:
        # 使用当前文件中的特征列
        features_df = df[feature_cols].copy()
    
    # 如果challenge_id在特征中但需要分离，先提取它
    challenge_id_series = None
    if 'challenge_id' in features_df.columns and not exclude_id:
        challenge_id_series = features_df['challenge_id'].copy()
        # 从特征中移除challenge_id（如果exclude_id为False，说明我们需要保留它但需要分离）
        # 但如果exclude_id为False，说明我们不需要排除它，所以保留在特征中
        # 实际上，如果exclude_id=False，challenge_id应该在feature_cols中
        # 我们在这里不排除它，但在保存时可以选择是否保存
    
    print(f"\nExtracted feature count: {len(features_df.columns)}")
    print(f"Feature data shape: {features_df.shape}")
    
    # 显示特征列表（只显示前50个，避免输出过长） / Display feature list (only first 50, avoid long output)
    print("\nAll feature column names (first 50):")
    display_cols = list(features_df.columns)[:50]
    for i, col in enumerate(display_cols, 1):
        print(f"  {i:3d}. {col}")
    if len(features_df.columns) > 50:
        print(f"  ... (total {len(features_df.columns)} features)")
    
    # 保存为CSV（如果exclude_id=False且challenge_id在特征中，保存时不包括challenge_id） / Save as CSV (if exclude_id=False and challenge_id in features, don't include in save)
    # 因为challenge_id需要单独保存 / Because challenge_id needs to be saved separately
    save_df = features_df.copy()
    if 'challenge_id' in save_df.columns and challenge_id_series is not None:
        # challenge_id将被单独处理，所以从保存的特征中移除 / challenge_id will be handled separately, so remove from saved features
        save_df = save_df.drop(columns=['challenge_id'])
        feature_cols_for_save = [
            col for col in features_df.columns if col != 'challenge_id'
        ]
    else:
        feature_cols_for_save = list(features_df.columns)
    
    print(f"\nSaving to: {output_file}")
    ensure_dir_exists(output_file)
    save_df.to_csv(output_file, index=False)
    print(f"✓ Successfully saved {len(feature_cols_for_save)} features to {output_file}")
    
    # 显示一些统计信息 / Display some statistics
    print("\nData Statistics:")
    print(f"  Total rows: {len(features_df)}")
    print(f"  Total columns: {len(feature_cols_for_save)}")
    print(f"  Total missing values: {save_df.isnull().sum().sum()}")
    if len(feature_cols_for_save) > 0:
        missing_ratio = (save_df.isnull().sum().sum() /
                         (len(save_df) * len(feature_cols_for_save)) * 100)
        print(f"  Missing value ratio: {missing_ratio:.2f}%")

    # 显示每列的数据类型 / Display data type for each column
    print("\nData Type Statistics:")
    dtype_counts = save_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # ========== 特征工程：应用指南中的特征工程逻辑 ========== / Feature engineering: Apply feature engineering logic from guide
    print("\n" + "="*60)
    print("Applying Feature Engineering...")
    print("="*60)
    
    is_train = not exclude_label  # 如果有label列，说明是训练集 / If label column exists, it's training set
    features_df_engineered, transformers = engineer_features(
        features_df, 
        is_train=is_train
    )
    
    # 提取用户ID（用于GroupKFold） / Extract user ID (for GroupKFold)
    print("\nExtracting User ID (for GroupKFold validation)...")
    user_ids = extract_user_id(features_df_engineered)
    
    # 更新保存的DataFrame（包含工程特征） / Update saved DataFrame (includes engineered features)
    save_df_engineered = features_df_engineered.copy()
    if 'challenge_id' in save_df_engineered.columns and challenge_id_series is not None:
        save_df_engineered = save_df_engineered.drop(columns=['challenge_id'])
    
    # 保存工程后的特征 / Save engineered features
    print(f"\nSaving engineered features to: {output_file}")
    ensure_dir_exists(output_file)
    save_df_engineered.to_csv(output_file, index=False)
    print(f"✓ Successfully saved {len(save_df_engineered.columns)} features (including engineered features) to {output_file}")
    
    return features_df_engineered, list(save_df_engineered.columns), challenge_id_series, user_ids, transformers


def extract_labels_and_ids(input_file, output_labels_file=None, output_ids_file=None):
    """
    从训练集或测试集中提取标签和ID
    
    Args:
        input_file: 输入的JSONL文件路径
        output_labels_file: 输出标签文件路径（可选，仅用于训练集）
        output_ids_file: 输出ID文件路径（可选，仅用于测试集）
        
    Returns:
        labels: 标签Series（如果有）或None
        ids: challenge_id Series（如果有）或None
    """
    print(f"\nExtracting labels and IDs: {input_file}")
    
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError:
                continue
    
    df = json_normalize(records)
    
    labels = None
    ids = None
    
    if 'label' in df.columns:
        labels = df['label']
        if output_labels_file:
            ensure_dir_exists(output_labels_file)
            labels.to_csv(output_labels_file, index=False, header=['label'])
            print(f"✓ Labels saved to: {output_labels_file}")
    
    if 'challenge_id' in df.columns:
        ids = df['challenge_id']
        if output_ids_file:
            ensure_dir_exists(output_ids_file)
            ids.to_csv(output_ids_file, index=False, header=['challenge_id'])
            print(f"✓ IDs saved to: {output_ids_file}")
    
    return labels, ids


def main():
    """
    主函数：支持命令行参数处理训练集和测试集
    确保训练集和测试集有统一的特征列
    """
    import os
    
    # 获取项目根目录（从 src/xgboost 向上两级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 切换到项目根目录
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        parser = argparse.ArgumentParser(description='Extract features from JSONL files')
        parser.add_argument(
            '--train-file', type=str, default='data/train.jsonl',
            help='Training set JSONL file path'
        )
        parser.add_argument(
            '--test-file', type=str, default='data/kaggle_test.jsonl',
            help='Test set JSONL file path'
        )
        parser.add_argument(
            '--train-output', type=str,
            default='cache/extracted_features.csv',
            help='Training set feature output file path'
        )
        parser.add_argument(
            '--test-output', type=str,
            default='cache/test_extracted_features.csv',
            help='Test set feature output file path'
        )
        parser.add_argument(
            '--train-labels-output', type=str,
            default='cache/train_labels.csv',
            help='Training set label output file path'
        )
        parser.add_argument(
            '--test-ids-output', type=str, default='cache/test_ids.csv',
            help='Test set ID output file path'
        )
        parser.add_argument('--train-only', action='store_true',
                            help='Process training set only')
        parser.add_argument('--test-only', action='store_true',
                            help='Process test set only')
        
        args = parser.parse_args()
        
        print(f"Project root directory: {project_root}")
        print("="*60)
        print("Extracting Training and Test Set Features (Unified Feature Columns)")
        print("="*60)
        
        # 第一步：确定统一的特征列列表 / Step 1: Determine unified feature column list
        unified_columns = None
        if not args.train_only and not args.test_only:
            # 同时处理训练集和测试集时，先确定统一的特征列 / When processing both training and test sets, first determine unified feature columns
            train_file_for_scan = (
                args.train_file if not args.test_only else None
            )
            test_file_for_scan = (
                args.test_file if not args.train_only else None
            )
            unified_columns = get_all_feature_columns(
                train_file_for_scan, test_file_for_scan
            )
            if unified_columns:
                print(f"\n✓ Unified feature column list determined (total {len(unified_columns)} columns)")
        
        # 处理训练集 / Process training set
        if not args.test_only:
            try:
                print("\n[Processing Training Set]")
                features_df, feature_cols, _, user_ids, transformers = extract_all_features(
                    args.train_file,
                    args.train_output,
                    exclude_label=True,  # 排除label列（只提取特征）
                    exclude_id=True,     # 排除challenge_id列
                    unified_feature_columns=unified_columns
                )
                
                # 保存用户ID（用于GroupKFold） / Save user IDs (for GroupKFold)
                if user_ids is not None:
                    user_ids_path = 'cache/train_user_ids.csv'
                    ensure_dir_exists(user_ids_path)
                    user_ids_df = pd.DataFrame({'user_id': user_ids})
                    user_ids_df.to_csv(user_ids_path, index=False)
                    print(f"✓ User IDs saved to: {user_ids_path}")
                
                # 提取并保存标签 / Extract and save labels
                labels, _ = extract_labels_and_ids(
                    args.train_file,
                    output_labels_file=args.train_labels_output
                )
                
                print("\n✓ Training set processing completed!")
                print(f"  Feature file: {args.train_output}")
                if labels is not None:
                    print(f"  Label file: {args.train_labels_output}")
                
                # 如果还没有统一的列列表，使用训练集的列作为统一列表 / If no unified column list yet, use training set columns as unified list
                if unified_columns is None:
                    unified_columns = feature_cols
            except FileNotFoundError:
                print(f"Error: Training set file not found: {args.train_file}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        # 处理测试集 / Process test set
        if not args.train_only:
            try:
                print("\n[Processing Test Set]")
                # 注意：测试集需要使用训练集的transformers（特别是source_encoder） / Note: Test set needs to use training set transformers (especially source_encoder)
                # 这里暂时不传递transformers，在实际使用时需要从训练集加载 / Here temporarily not passing transformers, need to load from training set in actual use
                features_df, feature_cols, challenge_id_series, _, _ = (
                    extract_all_features(
                        args.test_file,
                        args.test_output,
                        exclude_label=True,  # 测试集没有label列 / Test set has no label column
                        exclude_id=False,    # 保留challenge_id（需要分离） / Keep challenge_id (needs separation)
                        unified_feature_columns=unified_columns
                    )
                )
                
                # 保存challenge_id / Save challenge_id
                if challenge_id_series is not None:
                    ensure_dir_exists(args.test_ids_output)
                    ids_df = pd.DataFrame(
                        {'challenge_id': challenge_id_series}
                    )
                    ids_df.to_csv(args.test_ids_output, index=False)
                    print(
                        f"✓ challenge_id saved to: {args.test_ids_output}"
                    )
                
                print("\n✓ Test set processing completed!")
                print(f"  Feature file: {args.test_output}")
                if challenge_id_series is not None:
                    print(f"  ID file: {args.test_ids_output}")
            except FileNotFoundError:
                print(f"Error: Test set file not found: {args.test_file}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("All Extraction Completed!")
        print("="*60)
        if unified_columns:
            print(f"Training and test sets unified to {len(unified_columns)} feature columns")
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

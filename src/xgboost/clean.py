"""
数据清洗和降维脚本 / Data cleaning and dimensionality reduction script
对提取的特征进行清洗、特征工程和降维处理 / Clean, feature engineering and dimensionality reduction for extracted features
适用于Kaggle Influencer预测竞赛 / For Kaggle Influencer prediction competition
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
# PCA降维已移除，保留原特征 / PCA dimensionality reduction removed, keep original features
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(input_file):
    """
    加载提取的特征数据 / Load extracted feature data
    
    Args:
        input_file: 输入的CSV文件路径 / Input CSV file path
        
    Returns:
        DataFrame: 加载的数据 / Loaded data
    """
    import os
    
    print(f"Loading data: {input_file}")
    
    # 检查文件是否存在 / Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File does not exist: {input_file}")
    
    # 获取文件大小 / Get file size
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")
    
    # 对于大文件（>100MB），直接使用分块读取以避免超时 / For large files (>100MB), use chunked reading to avoid timeout
    if file_size > 100:
        print("  File is large, using optimized chunked reading strategy...")
        print("  Step 1: Quickly read first few rows to determine column structure...")
        
        # 尝试使用C引擎读取（快速），如果超时则使用Python引擎（稳定但慢） / Try C engine (fast), if timeout use Python engine (stable but slow)
        print("  Attempting to read with C engine...")
        use_python_engine = False
        
        # 先尝试读取少量行来确定列结构 / Try reading a few rows first to determine column structure
        try:
            print("    Step 1a: Reading first 10 rows to determine column structure...")
            # 使用更小的样本和更短的超时 / Use smaller sample and shorter timeout
            sample = pd.read_csv(input_file, nrows=10, low_memory=False, engine='c')
            print(f"  Success! Column count: {len(sample.columns)}")
            
            # 为object类型列创建dtype字典 / Create dtype dictionary for object type columns
            dtype_dict = {}
            for col in sample.columns:
                if sample[col].dtype == 'object':
                    dtype_dict[col] = 'object'
            
            print(f"  Step 2: Using C engine chunked reading (fast mode)...")
            chunk_list = []
            chunk_size = 10000  # 使用更大的chunk size以减少迭代次数 / Use larger chunk size to reduce iterations
            total_rows = 0
            
            reader = pd.read_csv(
                input_file, 
                chunksize=chunk_size, 
                low_memory=False,
                dtype=dtype_dict,
                engine='c'
            )
            
            for i, chunk in enumerate(reader):
                chunk_list.append(chunk)
                total_rows += len(chunk)
                if (i + 1) % 10 == 0:
                    print(f"    Read {total_rows:,} rows...")
            
            print(f"  Step 3: Merging all chunks...")
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"  Data shape: {df.shape}")
            print(f"  Column count: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            return df
            
        except (TimeoutError, Exception) as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                print(f"  C engine read timeout: {e}")
            else:
                print(f"  C engine read failed: {e}")
            print("  Switching to Python engine (slower but more stable, no timeout)...")
            use_python_engine = True
        
        # 如果C引擎失败，使用Python引擎 / If C engine fails, use Python engine
        if use_python_engine:
            print("  Using Python engine chunked reading (stable mode)...")
            try:
                # 先读取前几行确定列结构，使用更稳健的方法 / Read first few rows to determine column structure, use more robust method
                print("    Attempting to read sample rows...")
                try:
                    sample = pd.read_csv(input_file, nrows=5, engine='python', low_memory=False)
                    print(f"  Successfully read sample, column count: {len(sample.columns)}")
                except pd.errors.EmptyDataError:
                    # 如果读取样本失败，尝试直接读取整个文件 / If sample read fails, try reading entire file directly
                    print("  Sample read failed, attempting to read entire file directly...")
                    df = pd.read_csv(input_file, engine='python', low_memory=False)
                    print(f"  Success! Data shape: {df.shape}")
                    return df
                
                chunk_list = []
                chunk_size = 10000  # 使用更大的chunk size / Use larger chunk size
                total_rows = 0
                
                # 使用try-except处理可能的空chunk / Use try-except to handle possible empty chunks
                print("    Starting chunked reading...")
                reader = pd.read_csv(
                    input_file,
                    chunksize=chunk_size,
                    engine='python',
                    low_memory=False
                )
                
                chunk_count = 0
                for i, chunk in enumerate(reader):
                    if len(chunk) > 0:  # 只处理非空chunk / Only process non-empty chunks
                        chunk_list.append(chunk)
                        total_rows += len(chunk)
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(f"    Read {total_rows:,} rows...")
                    else:
                        print(f"    Warning: Chunk {i+1} is empty, skipping")
                        break  # 如果遇到空chunk，可能已经读完 / If empty chunk encountered, may have finished reading
                
                if len(chunk_list) == 0:
                    raise ValueError("No data read")
                
                print(f"  Merging all chunks...")
                df = pd.concat(chunk_list, ignore_index=True)
                print(f"  Data shape: {df.shape}")
                print(f"  Column count: {len(df.columns)}")
                return df
            except pd.errors.EmptyDataError as e:
                print(f"  Error: File is empty or cannot be parsed: {e}")
                # 尝试直接读取整个文件 / Try reading entire file directly
                print("  Attempting to read entire file directly...")
                try:
                    df = pd.read_csv(input_file, engine='python', low_memory=False)
                    print(f"  Success! Data shape: {df.shape}")
                    return df
                except Exception as e2:
                    raise ValueError(f"Cannot read file: {e2}")
            except Exception as e:
                print(f"  Python engine read failed: {e}")
                # 最后尝试：使用更保守的设置 / Last attempt: use more conservative settings
                print("  Attempting to read with more conservative settings...")
                try:
                    # 尝试不同的参数组合 / Try different parameter combinations
                    read_kwargs = {
                        'engine': 'python',
                        'low_memory': False,
                        'sep': ','
                    }
                    # 对于pandas >= 1.3.0，使用on_bad_lines / For pandas >= 1.3.0, use on_bad_lines
                    try:
                        df = pd.read_csv(input_file, **read_kwargs, on_bad_lines='skip')
                    except TypeError:
                        # 对于旧版本pandas，使用error_bad_lines=False / For older pandas, use error_bad_lines=False
                        try:
                            df = pd.read_csv(input_file, **read_kwargs, error_bad_lines=False)
                        except TypeError:
                            # 如果都不支持，直接读取 / If neither supported, read directly
                            df = pd.read_csv(input_file, **read_kwargs)
                    print(f"  Success! Data shape: {df.shape}")
                    return df
                except Exception as e2:
                    raise ValueError(f"All read methods failed: {e2}")
    else:
        # 对于小文件，直接读取 / For small files, read directly
        print("  Starting to read CSV file...")
        try:
            df = pd.read_csv(input_file, low_memory=False)
            print(f"  Data shape: {df.shape}")
            print(f"  Column count: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            return df
        except Exception as e:
            print(f"  Direct read failed: {e}")
            print("  Attempting chunked reading...")
            # 如果直接读取失败，尝试分块读取 / If direct read fails, try chunked reading
            chunk_list = []
            chunk_size = 10000
            for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
                chunk_list.append(chunk)
                if (i + 1) % 10 == 0:
                    print(f"    Read {((i + 1) * chunk_size):,} rows...")
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"  Data shape: {df.shape}")
            print(f"  Column count: {len(df.columns)}")
            return df


def analyze_data(df):
    """
    分析数据的基本统计信息 / Analyze basic statistical information of data
    
    Args:
        df: 输入DataFrame / Input DataFrame
    """
    print("\n" + "="*60)
    print("Basic Data Information Analysis")
    print("="*60)
    
    print(f"\nData shape: {df.shape}")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    
    # 数据类型统计 / Data type statistics
    print("\nData Type Statistics:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # 缺失值统计 / Missing value statistics
    print("\nMissing Value Statistics:")
    missing_counts = df.isnull().sum()
    missing_ratio = (missing_counts / len(df)) * 100
    missing_info = pd.DataFrame({
        'missing_count': missing_counts,
        'missing_ratio(%)': missing_ratio
    })
    missing_info = missing_info[missing_info['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_info) > 0:
        print(f"  Columns with missing values: {len(missing_info)}")
        print(f"  Total missing values: {missing_counts.sum()}")
        print(f"  Total missing value ratio: {(missing_counts.sum() / (len(df) * len(df.columns))) * 100:.2f}%")
        print("\n  Top 10 columns with most missing values:")
        for idx, row in missing_info.head(10).iterrows():
            print(f"    {idx}: {row['missing_count']} ({row['missing_ratio(%)']:.2f}%)")
    else:
        print("  No missing values")
    
    # 数值特征统计 / Numeric feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        print(f"\nNumeric Feature Statistics (total {len(numeric_cols)} columns):")
        print(f"  Mean range: [{df[numeric_cols].mean().min():.2f}, {df[numeric_cols].mean().max():.2f}]")
        print(f"  Std range: [{df[numeric_cols].std().min():.2f}, {df[numeric_cols].std().max():.2f}]")
        
        # 检查异常值（使用IQR方法） / Check outliers (using IQR method)
        print("\n  Outlier Detection (using IQR method):")
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if len(outlier_counts) > 0:
            print(f"    Columns with outliers: {len(outlier_counts)}")
            sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"    Top 5 columns with most outliers:")
            for col, count in sorted_outliers[:5]:
                print(f"      {col}: {count} outliers")
        else:
            print("    No significant outliers detected")
    
    # 分类特征统计 / Categorical feature statistics
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(categorical_cols) > 0:
        print(f"\nCategorical Feature Statistics (total {len(categorical_cols)} columns):")
        unique_counts = df[categorical_cols].nunique()
        print(f"  Unique value count range: [{unique_counts.min()}, {unique_counts.max()}]")
        print(f"  Top 5 columns with most unique values:")
        for col in unique_counts.nlargest(5).index:
            print(f"    {col}: {unique_counts[col]} unique values")


def remove_duplicates(df):
    """
    移除重复行 / Remove duplicate rows
    
    Args:
        df: 输入DataFrame / Input DataFrame
        
    Returns:
        DataFrame: 去重后的DataFrame / DataFrame after deduplication
    """
    print("\n" + "="*60)
    print("Removing Duplicates")
    print("="*60)
    
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    removed_rows = initial_rows - len(df_cleaned)
    
    print(f"  Original rows: {initial_rows}")
    print(f"  Rows after deduplication: {len(df_cleaned)}")
    print(f"  Removed duplicate rows: {removed_rows}")
    
    return df_cleaned


def handle_missing_values(df, strategy='nan'):
    """
    处理缺失值 / Handle missing values
    
    根据指南，XGBoost可以原生处理np.nan，会自动学习处理缺失值的最佳分裂方向 / According to guide, XGBoost can natively handle np.nan, automatically learns best split direction for missing values
    因此默认策略是保留np.nan，而不是填充 / Therefore default strategy is to keep np.nan, not fill
    
    Args:
        df: 输入DataFrame / Input DataFrame
        strategy: 填充策略 ('nan', 'median', 'mean', 'mode', 'drop') / Filling strategy ('nan', 'median', 'mean', 'mode', 'drop')
                  'nan'表示保留np.nan（XGBoost推荐） / 'nan' means keep np.nan (XGBoost recommended)
        
    Returns:
        DataFrame: 处理后的DataFrame / Processed DataFrame
    """
    print("\n" + "="*60)
    print(f"Handling Missing Values (strategy: {strategy})")
    print("="*60)
    
    df_cleaned = df.copy()
    
    # 分离数值列和非数值列 / Separate numeric and non-numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")
    
    if strategy == 'nan':
        # 保留np.nan，让XGBoost处理（推荐策略） / Keep np.nan, let XGBoost handle (recommended strategy)
        # 只统一转换None、'None'、空字符串等为np.nan / Only uniformly convert None, 'None', empty strings, etc. to np.nan
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].replace([None, 'None', ''], np.nan)
            elif df_cleaned[col].dtype in ['int64', 'float64']:
                df_cleaned[col] = df_cleaned[col].replace([None], np.nan)
        print(f"  Keeping np.nan for XGBoost native processing (recommended)")
        print(f"  Total missing values: {df_cleaned.isnull().sum().sum()}")
    elif strategy == 'drop':
        # 删除包含缺失值的行 / Delete rows containing missing values
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        removed_rows = initial_rows - len(df_cleaned)
        print(f"  Deleted rows: {removed_rows}")
    else:
        # 对数值列使用指定策略填充 / Fill numeric columns using specified strategy
        if len(numeric_cols) > 0:
            # 确保只处理实际存在的列 / Ensure only processing existing columns
            existing_numeric_cols = [col for col in numeric_cols if col in df_cleaned.columns]
            if len(existing_numeric_cols) > 0:
                # 分离全部缺失的列和部分缺失的列 / Separate fully missing columns and partially missing columns
                all_missing_cols = []
                partial_missing_cols = []
                
                for col in existing_numeric_cols:
                    missing_count = df_cleaned[col].isnull().sum()
                    if missing_count == len(df_cleaned):
                        # 全部缺失，用0填充 / Fully missing, fill with 0
                        all_missing_cols.append(col)
                    elif missing_count > 0:
                        # 部分缺失，用imputer填充 / Partially missing, fill with imputer
                        partial_missing_cols.append(col)
                    # 没有缺失的列不需要处理 / Columns without missing values don't need processing
                
                # 处理全部缺失的列 / Process fully missing columns
                if len(all_missing_cols) > 0:
                    df_cleaned[all_missing_cols] = 0
                    print(f"  Fully missing columns filled with 0: {len(all_missing_cols)} columns")
                
                # 处理部分缺失的列 / Process partially missing columns
                if len(partial_missing_cols) > 0:
                    if strategy == 'median':
                        imputer = SimpleImputer(strategy='median')
                    elif strategy == 'mean':
                        imputer = SimpleImputer(strategy='mean')
                    else:
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    
                    # fit_transform 返回 numpy 数组，需要转换回 DataFrame / fit_transform returns numpy array, need to convert back to DataFrame
                    imputed_data = imputer.fit_transform(df_cleaned[partial_missing_cols])
                    df_cleaned[partial_missing_cols] = pd.DataFrame(
                        imputed_data, 
                        columns=partial_missing_cols, 
                        index=df_cleaned.index
                    )
                    print(f"  Partially missing columns filled (strategy: {strategy}): {len(partial_missing_cols)} columns")
                
                print(f"  Numeric column filling complete, processed {len(existing_numeric_cols)} columns")
            else:
                print(f"  Warning: No valid numeric columns found")
        
        # 对分类列使用众数或0填充 / Fill categorical columns with mode or 0
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(0)
            print(f"  Categorical column filling complete")
    
    # 验证 / Verification
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"  Remaining missing values: {remaining_missing}")
    
    return df_cleaned


def handle_outliers(df, method='clip'):
    """
    处理异常值 / Handle outliers
    
    Args:
        df: 输入DataFrame / Input DataFrame
        method: 处理方法 ('clip', 'remove', 'none') / Processing method ('clip', 'remove', 'none')
        
    Returns:
        DataFrame: 处理后的DataFrame / Processed DataFrame
    """
    print("\n" + "="*60)
    print(f"Handling Outliers (method: {method})")
    print("="*60)
    
    if method == 'none':
        print("  Skipping outlier processing")
        return df
    
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'clip':
        # 使用IQR方法裁剪异常值 / Use IQR method to clip outliers
        clipped_count = 0
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before_clip = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            clipped_count += before_clip
        
        print(f"  Total clipped outliers: {clipped_count}")
    
    elif method == 'remove':
        # 删除包含异常值的行 / Delete rows containing outliers
        initial_rows = len(df_cleaned)
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        removed_rows = initial_rows - len(df_cleaned)
        print(f"  Deleted rows: {removed_rows}")
    
    return df_cleaned


def remove_low_variance_features(df, threshold=0.01):
    """
    移除低方差特征 / Remove low variance features
    
    Args:
        df: 输入DataFrame / Input DataFrame
        threshold: 方差阈值 / Variance threshold
        
    Returns:
        DataFrame: 处理后的DataFrame / Processed DataFrame
        list: 被移除的列名列表 / List of removed column names
    """
    print("\n" + "="*60)
    print(f"Removing Low Variance Features (threshold: {threshold})")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("  No numeric features, skipping")
        return df, []
    
    # 计算方差 / Calculate variance
    variances = df[numeric_cols].var()
    low_variance_cols = variances[variances < threshold].index.tolist()
    
    if len(low_variance_cols) > 0:
        print(f"  Removed {len(low_variance_cols)} low variance features")
        df_cleaned = df.drop(columns=low_variance_cols)
        return df_cleaned, low_variance_cols
    else:
        print(f"  No low variance features found")
        return df, []


def remove_highly_correlated_features(df, threshold=0.95):
    """
    移除高度相关的特征 / Remove highly correlated features
    
    Args:
        df: 输入DataFrame / Input DataFrame
        threshold: 相关系数阈值 / Correlation coefficient threshold
        
    Returns:
        DataFrame: 处理后的DataFrame / Processed DataFrame
        list: 被移除的列名列表 / List of removed column names
    """
    print("\n" + "="*60)
    print(f"Removing Highly Correlated Features (threshold: {threshold})")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("  Insufficient numeric features, skipping")
        return df, []
    
    # 计算相关系数矩阵 / Calculate correlation coefficient matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # 找到高度相关的特征对 / Find highly correlated feature pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 找出需要移除的列 / Find columns to remove
    to_remove = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > threshold)]
    
    if len(to_remove) > 0:
        print(f"  Removed {len(to_remove)} highly correlated features")
        df_cleaned = df.drop(columns=to_remove)
        return df_cleaned, to_remove
    else:
        print(f"  No highly correlated features found")
        return df, []


def engineer_features(df):
    """
    特征工程：创建关键比率特征（与 xgboost_baseline.py 一致） / Feature engineering: Create key ratio features (consistent with xgboost_baseline.py)
    这些特征对社交网络预测非常重要 / These features are very important for social network prediction
    
    Args:
        df: 输入DataFrame / Input DataFrame
        
    Returns:
        DataFrame: 添加了工程特征后的DataFrame / DataFrame with engineered features added
    """
    print("\n" + "="*60)
    print("Feature Engineering: Creating Ratio Features")
    print("="*60)
    
    df_engineered = df.copy()
    
    # 1. follower_following_ratio - 黄金特征
    # 优先使用主推文的用户特征，而不是quoted_status的用户特征
    user_followers_cols = [col for col in df.columns if 'followers_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not user_followers_cols:
        user_followers_cols = [col for col in df.columns if 'followers_count' in col.lower()]
    
    user_friends_cols = [col for col in df.columns if 'friends_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not user_friends_cols:
        user_friends_cols = [col for col in df.columns if 'friends_count' in col.lower()]
    
    if user_followers_cols and user_friends_cols:
        followers_col = user_followers_cols[0]
        friends_col = user_friends_cols[0]
        # 将相关列转换为数值类型（如果它们是字符串）
        followers_series = pd.to_numeric(df_engineered[followers_col], errors='coerce')
        friends_series = pd.to_numeric(df_engineered[friends_col], errors='coerce')
        # 只对非缺失值计算比率
        mask = followers_series.notna() & friends_series.notna()
        df_engineered['follower_following_ratio'] = 0.0
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            followers_series.loc[mask] / 
            (friends_series.loc[mask] + 1)
        )
        print(f"  ✓ Created: follower_following_ratio (using {followers_col} / {friends_col})")
        print(f"    Non-null ratio values: {mask.sum()}/{len(df_engineered)}")
    
    # 2. user_activity_rate
    user_statuses_cols = [col for col in df.columns if 'statuses_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not user_statuses_cols:
        user_statuses_cols = [col for col in df.columns if 'statuses_count' in col.lower()]
    
    user_created_cols = [col for col in df.columns if 'user.created_at' in col.lower() and 'quoted_status' not in col.lower()]
    if not user_created_cols:
        user_created_cols = [col for col in df.columns if 'created_at' in col.lower() and 'user' in col.lower()]
    
    if user_statuses_cols and user_created_cols:
        statuses_col = user_statuses_cols[0]
        created_col = user_created_cols[0]
        
        # 计算账户年龄 / Calculate account age
        def calculate_account_age(created_at_str):
            if pd.isna(created_at_str):
                return None
            try:
                # Twitter created_at格式: "Wed Mar 17 12:59:53 +0000 2021" / Twitter created_at format: "Wed Mar 17 12:59:53 +0000 2021"
                created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
                now = datetime.now(created_at.tzinfo)
                return (now - created_at).days
            except:
                return None
        
        account_ages = df_engineered[created_col].apply(calculate_account_age)
        df_engineered['account_age_days'] = account_ages
        
        # 将statuses_col转换为数值类型（如果它是字符串） / Convert statuses_col to numeric type (if it's a string)
        statuses_series = pd.to_numeric(df_engineered[statuses_col], errors='coerce')
        account_age_series = pd.to_numeric(df_engineered['account_age_days'], errors='coerce')
        
        # 计算活动率（只对有效值计算） / Calculate activity rate (only for valid values)
        mask = statuses_series.notna() & account_age_series.notna()
        df_engineered['user_activity_rate'] = 0.0
        df_engineered.loc[mask, 'user_activity_rate'] = (
            statuses_series.loc[mask] / 
            (account_age_series.loc[mask] + 1)
        )
        print(f"  ✓ Created: user_activity_rate (using {statuses_col} / account_age_days)")
        print(f"    Non-null activity rate values: {mask.sum()}/{len(df_engineered)}")
    
    # 3. tweet_engagement_rate
    retweet_cols = [col for col in df.columns if 'retweet_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not retweet_cols:
        retweet_cols = [col for col in df.columns if 'retweet_count' in col.lower()]
    
    favorite_cols = [col for col in df.columns if 'favorite_count' in col.lower() and 'quoted_status' not in col.lower()]
    if not favorite_cols:
        favorite_cols = [col for col in df.columns if 'favorite_count' in col.lower()]
    
    if retweet_cols and favorite_cols and user_followers_cols:
        retweet_col = retweet_cols[0]
        favorite_col = favorite_cols[0]
        followers_col = user_followers_cols[0]
        
        # 将相关列转换为数值类型（如果它们是字符串） / Convert related columns to numeric type (if they are strings)
        retweet_series = pd.to_numeric(df_engineered[retweet_col], errors='coerce')
        favorite_series = pd.to_numeric(df_engineered[favorite_col], errors='coerce')
        followers_series = pd.to_numeric(df_engineered[followers_col], errors='coerce')
        
        # 只对有效值计算参与率 / Only calculate engagement rate for valid values
        mask = (retweet_series.notna() & 
                favorite_series.notna() & 
                followers_series.notna())
        df_engineered['tweet_engagement_rate'] = 0.0
        df_engineered.loc[mask, 'tweet_engagement_rate'] = (
            (retweet_series.loc[mask] + 
             favorite_series.loc[mask]) / 
            (followers_series.loc[mask] + 1)
        )
        print(f"  ✓ Created: tweet_engagement_rate")
        print(f"    Non-null engagement rate values: {mask.sum()}/{len(df_engineered)}")
    
    print(f"  Feature engineering complete, new features: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered


def encode_categorical_features(df, use_ordinal=True):
    """
    编码分类特征为数值 / Encode categorical features to numeric
    
    根据指南，对于高基数分类特征（如source），使用OrdinalEncoder / According to guide, for high cardinality categorical features (like source), use OrdinalEncoder
    XGBoost可以高效处理Ordinal编码的整数 / XGBoost can efficiently handle Ordinal encoded integers
    
    Args:
        df: 输入DataFrame / Input DataFrame
        use_ordinal: 是否使用OrdinalEncoder（推荐，适合XGBoost） / Whether to use OrdinalEncoder (recommended, suitable for XGBoost)
        
    Returns:
        DataFrame: 编码后的DataFrame / Encoded DataFrame
    """
    from sklearn.preprocessing import OrdinalEncoder
    
    print("\n" + "="*60)
    print("Encoding Categorical Features")
    print("="*60)
    
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("  No categorical features, skipping")
        return df_encoded
    
    print(f"  Categorical features to encode: {len(categorical_cols)}")
    
    if use_ordinal:
        # 使用OrdinalEncoder（推荐，适合XGBoost） / Use OrdinalEncoder (recommended, suitable for XGBoost)
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # 准备数据：将NaN转换为字符串'unknown' / Prepare data: convert NaN to string 'unknown'
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            col_data = df_encoded[col].fillna('unknown').values.reshape(-1, 1)
            
            encoded_data = encoder.fit_transform(col_data)
            df_encoded[col] = encoded_data.flatten()
            
            print(f"    {col}: Ordinal encoding ({unique_count} unique values)")
    else:
        # 旧方法：根据唯一值数量选择编码方式 / Old method: choose encoding based on unique value count
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            
            if unique_count < 10 and unique_count > 1:
                # 使用独热编码 / Use one-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                print(f"    {col}: One-hot encoding ({unique_count} unique values)")
            else:
                # 使用标签编码（转换为数值） / Use label encoding (convert to numeric)
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                print(f"    {col}: Label encoding ({unique_count} unique values)")
    
    print(f"  Encoding complete, current column count: {len(df_encoded.columns)}")
    
    return df_encoded


def normalize_features(df, method='standard'):
    """
    标准化/归一化特征 / Normalize/standardize features
    
    Args:
        df: 输入DataFrame / Input DataFrame
        method: 标准化方法 ('standard', 'robust', 'none') / Normalization method ('standard', 'robust', 'none')
        
    Returns:
        DataFrame: 标准化后的DataFrame / Normalized DataFrame
        scaler: 使用的scaler对象（用于后续数据转换） / Scaler object used (for subsequent data transformation)
    """
    print("\n" + "="*60)
    print(f"Feature Normalization (method: {method})")
    print("="*60)
    
    if method == 'none':
        print("  Skipping normalization")
        return df, None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("  No numeric features, skipping")
        return df, None
    
    df_normalized = df.copy()
    
    # 确保只处理实际存在的列 / Ensure only processing existing columns
    existing_numeric_cols = [col for col in numeric_cols if col in df_normalized.columns]
    
    if len(existing_numeric_cols) == 0:
        print("  No valid numeric features found, skipping normalization")
        return df, None
    
    if method == 'standard':
        scaler = StandardScaler()
        print(f"  Using StandardScaler to normalize {len(existing_numeric_cols)} numeric features")
    elif method == 'robust':
        scaler = RobustScaler()
        print(f"  Using RobustScaler to normalize {len(existing_numeric_cols)} numeric features")
    else:
        print(f"  Unknown method, skipping normalization")
        return df, None
    
    # fit_transform 返回 numpy 数组，需要转换回 DataFrame / fit_transform returns numpy array, need to convert back to DataFrame
    normalized_data = scaler.fit_transform(df_normalized[existing_numeric_cols])
    df_normalized[existing_numeric_cols] = pd.DataFrame(
        normalized_data,
        columns=existing_numeric_cols,
        index=df_normalized.index
    )
    print(f"  Normalization complete")
    
    return df_normalized, scaler


def reduce_dimensionality(df, method='none', n_components=None, variance_ratio=0.95):
    """
    降维处理（已禁用，保留原特征） / Dimensionality reduction (disabled, keep original features)
    
    Args:
        df: 输入DataFrame / Input DataFrame
        method: 降维方法（已禁用，始终返回原数据） / Dimensionality reduction method (disabled, always returns original data)
        n_components: 主成分数量（已禁用） / Number of principal components (disabled)
        variance_ratio: 保留的方差比例（已禁用） / Variance ratio to retain (disabled)
        
    Returns:
        DataFrame: 原始DataFrame（不进行降维） / Original DataFrame (no dimensionality reduction)
        transformer: None（降维已禁用） / None (dimensionality reduction disabled)
    """
    print("\n" + "="*60)
    print(f"Dimensionality Reduction (method: {method})")
    print("="*60)
    
    print("  Dimensionality reduction disabled, keeping all original features")
    return df, None


def clean_and_reduce(input_file, output_file='cache/cleaned_features.csv',
                     missing_strategy='median',
                     outlier_method='clip',
                     remove_low_var=False,  # 默认禁用，对XGBoost树模型通常不需要
                     remove_high_corr=False,  # 默认禁用，或使用更高阈值如0.99
                     normalize_method='none',  # 默认禁用，XGBoost树模型不需要标准化
                     reduce_method='none',
                     variance_ratio=0.95,
                     test_input_file=None,
                     test_output_file=None):
    """
    完整的数据清洗流程（支持训练集和测试集，已移除PCA降维）
    
    Args:
        input_file: 训练集输入CSV文件路径
        output_file: 训练集输出CSV文件路径
        missing_strategy: 缺失值处理策略
        outlier_method: 异常值处理方法
        remove_low_var: 是否移除低方差特征
        remove_high_corr: 是否移除高度相关特征
        normalize_method: 标准化方法
        reduce_method: 降维方法（已禁用，保留原特征）
        variance_ratio: PCA保留的方差比例（已禁用）
        test_input_file: 测试集输入CSV文件路径（可选）
        test_output_file: 测试集输出CSV文件路径（可选）
        
    Returns:
        DataFrame: 清洗后的训练集DataFrame（保留原特征）
        dict: 包含fitted transformers的字典（用于测试集转换）
    """
    print("="*60)
    print("Data Cleaning Pipeline (Keep Original Features)")
    print("="*60)
    
    transformers = {}
    
    # ========== 训练集处理 ========== / Training set processing
    print("\n[Processing Training Set]")
    # 1. 加载数据 / Load data
    df_train = load_data(input_file)
    
    # ========== 如果提供了测试集，先对齐特征列 ========== / If test set provided, align feature columns first
    if test_input_file and test_output_file:
        import os
        if os.path.exists(test_input_file):
            print("\n[Pre-aligning Feature Columns]")
            print("  Loading test set to align feature columns...")
            df_test_preview = load_data(test_input_file)
            
            # 获取训练集和测试集的所有列 / Get all columns from training and test sets
            train_cols = set(df_train.columns)
            test_cols = set(df_test_preview.columns)
            
            # 找出差异 / Find differences
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            common_cols = train_cols & test_cols
            
            print(f"  Training set columns: {len(train_cols)}")
            print(f"  Test set columns: {len(test_cols)}")
            print(f"  Common columns: {len(common_cols)}")
            
            if len(missing_in_test) > 0:
                print(f"  Missing columns in test set (will be added to test set, filled with NaN): {len(missing_in_test)}")
            if len(missing_in_train) > 0:
                print(f"  Missing columns in training set (will be added to training set, filled with NaN): {len(missing_in_train)}")
            
            # 确定统一的列列表（按字母顺序排序以确保一致性） / Determine unified column list (sorted alphabetically for consistency)
            unified_cols = sorted(list(train_cols | test_cols))
            
            # 对齐训练集 / Align training set
            for col in unified_cols:
                if col not in df_train.columns:
                    df_train[col] = None
            df_train = df_train[unified_cols]
            
            print(f"  ✓ Training set aligned to {len(unified_cols)} columns")
            
            # 保存统一列列表，用于后续测试集处理
            transformers['unified_columns'] = unified_cols
    
    # 2. 分析数据
    analyze_data(df_train)
    
    # 3. 移除重复值
    df_train = remove_duplicates(df_train)
    
    # 3.5. 特征工程：创建比率特征（在处理缺失值之前，需要原始数据）
    # 这必须在编码分类特征之前，因为需要原始的日期字符串和数值
    df_train = engineer_features(df_train)
    
    # 保存特征工程后的列名（用于测试集对齐）
    transformers['train_columns_after_feature_eng'] = df_train.columns.tolist()
    
    # 4. 处理缺失值（fit on train）
    imputer = None
    if missing_strategy != 'drop':
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        existing_numeric_cols = [col for col in numeric_cols if col in df_train.columns]
        
        if len(existing_numeric_cols) > 0:
            all_missing_cols = []
            partial_missing_cols = []
            
            for col in existing_numeric_cols:
                missing_count = df_train[col].isnull().sum()
                if missing_count == len(df_train):
                    all_missing_cols.append(col)
                elif missing_count > 0:
                    partial_missing_cols.append(col)
            
            if len(all_missing_cols) > 0:
                df_train[all_missing_cols] = 0
            
            if len(partial_missing_cols) > 0:
                if missing_strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif missing_strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                
                imputer.fit(df_train[partial_missing_cols])
                imputed_data = imputer.transform(df_train[partial_missing_cols])
                df_train[partial_missing_cols] = pd.DataFrame(
                    imputed_data,
                    columns=partial_missing_cols,
                    index=df_train.index
                )
                transformers['imputer'] = imputer
                transformers['imputer_cols'] = partial_missing_cols
        
        categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_modes = {}
        for col in categorical_cols:
            # 保存所有分类列的众数（即使没有缺失值），用于测试集处理
            mode_val = df_train[col].mode()
            if len(mode_val) > 0:
                mode_to_use = mode_val[0]
                categorical_modes[col] = mode_to_use
            else:
                categorical_modes[col] = 0
            
            # 如果训练集有缺失值，用众数填充
            if df_train[col].isnull().sum() > 0:
                df_train[col] = df_train[col].fillna(categorical_modes[col])
        transformers['categorical_modes'] = categorical_modes
    else:
        df_train = df_train.dropna()
    
    # 5. 处理异常值
    df_train = handle_outliers(df_train, method=outlier_method)
    
    # 6. 编码分类特征（fit on train，保存encoders）
    categorical_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()
    encoders = {}
    onehot_columns = {}  # 保存独热编码产生的列名
    for col in categorical_cols:
        unique_count = df_train[col].nunique()
        if unique_count < 10 and unique_count > 1:
            # 独热编码
            dummies = pd.get_dummies(df_train[col], prefix=col, drop_first=True)
            # 保存独热编码产生的列名
            onehot_columns[col] = dummies.columns.tolist()
            df_train = pd.concat([df_train.drop(columns=[col]), dummies], axis=1)
            encoders[col] = ('onehot', unique_count)
        else:
            # 标签编码
            df_train[col] = pd.Categorical(df_train[col]).codes
            encoders[col] = ('label', None)
    transformers['encoders'] = encoders
    transformers['onehot_columns'] = onehot_columns  # 保存独热编码列名
    
    # 7. 移除低方差特征（fit on train）
    removed_low_var_cols = []
    if remove_low_var:
        df_train, removed_low_var_cols = remove_low_variance_features(df_train, threshold=0.01)
        if len(removed_low_var_cols) > 0:
            print(f"  Removed low variance columns: {len(removed_low_var_cols)}")
    
    # 8. 移除高度相关特征（fit on train） / Remove highly correlated features (fit on train)
    removed_high_corr_cols = []
    if remove_high_corr:
        df_train, removed_high_corr_cols = remove_highly_correlated_features(df_train, threshold=0.95)
        if len(removed_high_corr_cols) > 0:
            print(f"  Removed highly correlated columns: {len(removed_high_corr_cols)}")
    
    # 保存要移除的列名 / Save column names to remove
    transformers['removed_cols'] = removed_low_var_cols + removed_high_corr_cols
    
    # 9. 标准化特征（fit on train） / Normalize features (fit on train)
    # 保存标准化前的列名（用于测试集标准化前的对齐） / Save column names before normalization (for test set pre-normalization alignment)
    transformers['train_columns_pre_normalize'] = df_train.columns.tolist()
    
    # 保存标准化前使用的数值列名 / Save numeric column names used before normalization
    numeric_cols_before_normalize = df_train.select_dtypes(include=[np.number]).columns.tolist()
    transformers['train_numeric_cols_for_scaler'] = numeric_cols_before_normalize
    
    df_train, scaler = normalize_features(df_train, method=normalize_method)
    transformers['scaler'] = scaler
    transformers['scaler_method'] = normalize_method
    
    # 10. 降维（已禁用，保留原特征） / Dimensionality reduction (disabled, keep original features)
    # PCA降维已移除，保留所有原始特征 / PCA dimensionality reduction removed, keep all original features
    if reduce_method != 'none':
        print("\nWarning: reduce_method is not 'none', but PCA dimensionality reduction has been removed, will keep all features")
    
    # 保存训练集的最终列顺序（用于测试集最终对齐） / Save final column order of training set (for test set final alignment)
    transformers['train_columns'] = df_train.columns.tolist()
    
    # 11. 最终统计 / Final statistics
    print("\n" + "="*60)
    print("Training Set Final Data Statistics")
    print("="*60)
    print(f"  Final data shape: {df_train.shape}")
    print(f"  Final column count: {len(df_train.columns)}")
    print(f"  Total missing values: {df_train.isnull().sum().sum()}")
    
    # 12. 保存训练集结果 / Save training set results
    print(f"\nSaving training set to: {output_file}")
    df_train.to_csv(output_file, index=False)
    print(f"✓ Successfully saved cleaned training set data to {output_file}")
    
    # ========== 测试集处理 ========== / Test set processing
    df_test = None
    if test_input_file and test_output_file:
        # 检查测试集输入文件是否存在 / Check if test set input file exists
        import os
        if not os.path.exists(test_input_file):
            print(f"\nWarning: Test set input file does not exist: {test_input_file}")
            print("Skipping test set processing. This is normal if you only want to process the training set.")
            print("If you want to process the test set, please ensure the file exists or use --test-input to specify the correct path.")
        else:
            print("\n" + "="*60)
            print("[Processing Test Set]")
            print("="*60)
            
            # 1. 加载测试集 / Load test set
            df_test = load_data(test_input_file)
            
            # 1.5. 对齐特征列（使用训练集的统一列列表） / Align feature columns (using training set unified column list)
            unified_cols = transformers.get('unified_columns')
            if unified_cols:
                print("\n  Aligning test set feature columns...")
                for col in unified_cols:
                    if col not in df_test.columns:
                        df_test[col] = None
                df_test = df_test[unified_cols]
                print(f"  ✓ Test set aligned to {len(unified_cols)} columns")
            
            # 2. 跳过移除重复值（测试集必须保持原始行数，以便与ID对应） / Skip removing duplicates (test set must keep original row count to match IDs)
            print("\n  Skipping duplicate removal (test set keeps original row count to match IDs)")
            initial_test_rows = len(df_test)
            
            # 2.5. 特征工程：创建比率特征（在处理缺失值之前，需要原始数据） / Feature engineering: Create ratio features (before handling missing values, need original data)
            # 这必须在编码分类特征之前，因为需要原始的日期字符串和数值 / This must be before encoding categorical features, because need original date strings and numeric values
            df_test = engineer_features(df_test)
            
            # 2.6. 对齐特征工程后的列（确保测试集有训练集的所有工程特征） / Align columns after feature engineering (ensure test set has all engineered features from training set)
            train_cols_after_fe = transformers.get('train_columns_after_feature_eng', [])
            if train_cols_after_fe:
                missing_cols = set(train_cols_after_fe) - set(df_test.columns)
                extra_cols = set(df_test.columns) - set(train_cols_after_fe)
                
                # 添加缺失的列（用0填充） / Add missing columns (fill with 0)
                if len(missing_cols) > 0:
                    for col in missing_cols:
                        df_test[col] = 0
                    print(f"  Post-feature-engineering alignment: Added {len(missing_cols)} missing columns")
                
                # 移除多余的列 / Remove extra columns
                if len(extra_cols) > 0:
                    df_test = df_test.drop(columns=list(extra_cols))
                    print(f"  Post-feature-engineering alignment: Removed {len(extra_cols)} extra columns")
                
                # 确保列顺序一致 / Ensure column order is consistent
                df_test = df_test[train_cols_after_fe]
                print(f"  ✓ Test set post-feature-engineering alignment: {len(train_cols_after_fe)} columns")
            
            # 3. 处理缺失值（使用训练集的imputer）
            if imputer is not None and 'imputer_cols' in transformers:
                imputer_cols = transformers['imputer_cols']
                # 处理全部缺失的列
                numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    if col in df_test.columns and df_test[col].isnull().sum() == len(df_test):
                        df_test[col] = 0
                
                # 使用训练集的imputer转换
                existing_imputer_cols = [col for col in imputer_cols if col in df_test.columns]
                if len(existing_imputer_cols) > 0:
                    imputed_data = imputer.transform(df_test[existing_imputer_cols])
                    df_test[existing_imputer_cols] = pd.DataFrame(
                        imputed_data,
                        columns=existing_imputer_cols,
                        index=df_test.index
                    )
            
            # 处理分类列的缺失值（使用训练集的众数）
            categorical_modes = transformers.get('categorical_modes', {})
            for col, mode_val in categorical_modes.items():
                if col in df_test.columns and df_test[col].isnull().sum() > 0:
                    df_test[col] = df_test[col].fillna(mode_val)
            
            # 4. 处理异常值（使用训练集的bounds）
            if outlier_method != 'none':
                # 对测试集使用相同的clip bounds（基于训练集统计）
                pass  # 跳过，因为clip需要重新计算bounds
            
            # 5. 编码分类特征（使用训练集的编码方式）
            encoders = transformers.get('encoders', {})
            onehot_columns = transformers.get('onehot_columns', {})
            for col, (encoding_type, unique_count) in encoders.items():
                if col in df_test.columns:
                    if encoding_type == 'onehot':
                        # 独热编码（需要对齐训练集的列）
                        # 只创建训练集中存在的独热编码列
                        train_onehot_cols = onehot_columns.get(col, [])
                        if len(train_onehot_cols) > 0:
                            # 为测试集创建所有训练集的独热编码列（初始化为0）
                            for onehot_col in train_onehot_cols:
                                df_test[onehot_col] = 0
                            
                            # 对测试集的值进行独热编码
                            dummies = pd.get_dummies(df_test[col], prefix=col, drop_first=True)
                            
                            # 确保 dummies 的索引与 df_test 匹配
                            # get_dummies 返回的 DataFrame 索引应该已经与输入匹配，但为了安全起见我们重新对齐
                            dummies = dummies.reindex(df_test.index, fill_value=0)
                            
                            # 只保留训练集中存在的列
                            for onehot_col in train_onehot_cols:
                                if onehot_col in dummies.columns:
                                    df_test[onehot_col] = dummies[onehot_col].values
                                # 如果测试集中没有这个类别，保持为0（已在上面的初始化中设置）
                            
                            # 删除原始列
                            df_test = df_test.drop(columns=[col])
                        else:
                            # 如果没有保存的训练集列，使用默认方式（不推荐）
                            dummies = pd.get_dummies(df_test[col], prefix=col, drop_first=True)
                            df_test = pd.concat([df_test.drop(columns=[col]), dummies], axis=1)
                    else:
                        # 标签编码
                        # 对于测试集中出现的新类别，会被编码为-1，我们将其设为0
                        codes = pd.Categorical(df_test[col]).codes
                        df_test[col] = codes
                        # 将-1（新类别）替换为0
                        df_test.loc[df_test[col] == -1, col] = 0
            
            # 6. 移除低方差和高度相关特征（使用训练集的列列表）
            removed_cols = transformers.get('removed_cols', [])
            for col in removed_cols:
                if col in df_test.columns:
                    df_test = df_test.drop(columns=[col])
            
        # 7. 对齐列（标准化前对齐，确保测试集有训练集标准化前的所有列）
        train_columns_pre_normalize = transformers.get('train_columns_pre_normalize')
        if train_columns_pre_normalize:
            missing_cols = set(train_columns_pre_normalize) - set(df_test.columns)
            extra_cols = set(df_test.columns) - set(train_columns_pre_normalize)
            
            # 添加缺失的列（用0填充）
            if len(missing_cols) > 0:
                for col in missing_cols:
                    df_test[col] = 0
                print(f"  Pre-normalization alignment: Added {len(missing_cols)} missing columns")
            
            # 移除多余的列（特别是独热编码产生的额外列） / Remove extra columns (especially those from one-hot encoding)
            if len(extra_cols) > 0:
                df_test = df_test.drop(columns=list(extra_cols))
                print(f"  Pre-normalization alignment: Removed {len(extra_cols)} extra columns")
            
            # 确保列顺序一致 / Ensure column order is consistent
            df_test = df_test[train_columns_pre_normalize]
            
            # 8. 标准化（使用训练集的scaler） / Normalization (using training set scaler)
            scaler = transformers.get('scaler')
            scaler_method = transformers.get('scaler_method', 'none')
            if scaler is not None and scaler_method != 'none':
                # 获取训练集标准化时使用的列名（使用scaler的实际特征名） / Get column names used during training set normalization (using scaler's actual feature names)
                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                    # 使用scaler实际拟合时的特征名（这是最准确的） / Use feature names from scaler's actual fit (most accurate)
                    train_scaler_numeric_cols = list(scaler.feature_names_in_)
                else:
                    # 如果没有feature_names_in_，使用保存的数值列名 / If no feature_names_in_, use saved numeric column names
                    train_scaler_numeric_cols = transformers.get('train_numeric_cols_for_scaler', [])
                    if not train_scaler_numeric_cols:
                        # 最后的备选方案：从保存的列名中推断 / Last fallback: infer from saved column names
                        train_columns_final = transformers.get('train_columns', [])
                        train_scaler_numeric_cols = train_columns_final
                
                # 确保测试集有训练集标准化时需要的所有列 / Ensure test set has all columns needed for training set normalization
                missing_cols = set(train_scaler_numeric_cols) - set(df_test.columns)
                extra_cols = set(df_test.columns) - set(train_scaler_numeric_cols)
                
                # 添加缺失的列（用0填充） / Add missing columns (fill with 0)
                if len(missing_cols) > 0:
                    for col in missing_cols:
                        df_test[col] = 0
                    print(f"  Pre-normalization: Added {len(missing_cols)} missing columns")
                
                # 移除多余的列（标准化前不需要的列，但保留它们用于后续对齐）
                # 注意：这里只移除独热编码产生的额外列，不删除所有多余列
                # 因为标准化后还需要对齐
                
                # 准备标准化用的列（只包含训练集标准化时使用的列，按相同顺序）
                df_test_for_scaler = df_test[train_scaler_numeric_cols].copy()
                
                # 只选择数值列进行标准化
                test_numeric_cols = df_test_for_scaler.select_dtypes(include=[np.number]).columns.tolist()
                
                # 确保测试集的数值列与训练集的完全匹配
                train_numeric_set = set(train_scaler_numeric_cols)
                test_numeric_set = set(test_numeric_cols)
                
                # 添加训练集中有但测试集中没有的数值列（用0填充）
                missing_numeric = train_numeric_set - test_numeric_set
                for col in missing_numeric:
                    df_test_for_scaler[col] = 0
                
                # 移除测试集中有但训练集中没有的数值列
                extra_numeric = test_numeric_set - train_numeric_set
                if len(extra_numeric) > 0:
                    df_test_for_scaler = df_test_for_scaler.drop(columns=list(extra_numeric))
                
                # 确保列顺序与训练集一致
                df_test_for_scaler = df_test_for_scaler[train_scaler_numeric_cols]
                
                if len(df_test_for_scaler.columns) > 0:
                    # 执行标准化
                    normalized_data = scaler.transform(df_test_for_scaler)
                    # 更新测试集中的数值列
                    for i, col in enumerate(df_test_for_scaler.columns):
                        df_test[col] = normalized_data[:, i]
            
            # 9. 最终对齐列（确保测试集与训练集列完全一致）
            train_columns = transformers.get('train_columns')
            if train_columns:
                missing_cols = set(train_columns) - set(df_test.columns)
                extra_cols = set(df_test.columns) - set(train_columns)
                
                # 添加缺失的列（用0填充）
                if len(missing_cols) > 0:
                    for col in missing_cols:
                        df_test[col] = 0
                    print(f"  Final alignment: Added {len(missing_cols)} missing columns")
                
                # 移除多余的列 / Remove extra columns
                if len(extra_cols) > 0:
                    df_test = df_test.drop(columns=list(extra_cols))
                    print(f"  Final alignment: Removed {len(extra_cols)} extra columns")
                
                # 确保列顺序一致 / Ensure column order is consistent
                df_test = df_test[train_columns]
                print(f"  ✓ Test set final alignment: {len(train_columns)} columns")
            
            # 10. 最终统计 / Final statistics
            print("\n" + "="*60)
            print("Test Set Final Data Statistics")
            print("="*60)
            print(f"  Final data shape: {df_test.shape}")
            print(f"  Final column count: {len(df_test.columns)}")
            print(f"  Total missing values: {df_test.isnull().sum().sum()}")
            
            # 11. 保存测试集结果 / Save test set results
            print(f"\nSaving test set to: {test_output_file}")
            df_test.to_csv(test_output_file, index=False)
            print(f"✓ Successfully saved cleaned test set data to {test_output_file}")
    
    return df_train, transformers, df_test


def main():
    """
    主函数
    """
    import argparse
    import os
    
    # 获取项目根目录（从 src/xgboost 向上两级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 切换到项目根目录
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        parser = argparse.ArgumentParser(description='Data cleaning and dimensionality reduction script')
        parser.add_argument('--input', type=str, default='cache/extracted_features.csv',
                            help='Training set input CSV file path')
        parser.add_argument('--output', type=str, default='cache/cleaned_features.csv',
                            help='Training set output CSV file path')
        parser.add_argument('--test-input', type=str, default='cache/test_extracted_features.csv',
                            help='Test set input CSV file path')
        parser.add_argument('--test-output', type=str, default='cache/test_cleaned_features.csv',
                            help='Test set output CSV file path')
        parser.add_argument('--missing-strategy', type=str, default='median',
                            choices=['median', 'mean', 'mode', 'drop'],
                            help='Missing value handling strategy')
        parser.add_argument('--outlier-method', type=str, default='clip',
                            choices=['clip', 'remove', 'none'],
                            help='Outlier handling method')
        parser.add_argument('--remove-low-var', action='store_true', default=False,
                            help='Remove low variance features (default: False, suitable for XGBoost)')
        parser.add_argument('--remove-high-corr', action='store_true', default=False,
                            help='Remove highly correlated features (default: False, suitable for XGBoost)')
        parser.add_argument('--normalize', type=str, default='none',
                            choices=['standard', 'robust', 'none'],
                            help='Normalization method (default: none, XGBoost tree models do not need normalization)')
        parser.add_argument('--reduce', type=str, default='none',
                            choices=['pca', 'none'],
                            help='Dimensionality reduction method (disabled, keep original features)')
        parser.add_argument('--variance-ratio', type=float, default=0.95,
                            help='PCA variance ratio to retain (disabled)')
        
        args = parser.parse_args()
        
        print(f"Project root directory: {project_root}")
        
        df_train, transformers, df_test = clean_and_reduce(
            input_file=args.input,
            output_file=args.output,
            missing_strategy=args.missing_strategy,
            outlier_method=args.outlier_method,
            remove_low_var=args.remove_low_var,
            remove_high_corr=args.remove_high_corr,
            normalize_method=args.normalize,
            reduce_method=args.reduce,
            variance_ratio=args.variance_ratio,
            test_input_file=args.test_input,
            test_output_file=args.test_output
        )
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()


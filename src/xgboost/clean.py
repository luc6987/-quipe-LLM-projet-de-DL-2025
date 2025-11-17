"""
数据清洗和降维脚本
对提取的特征进行清洗、特征工程和降维处理
适用于Kaggle Influencer预测竞赛
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
# PCA降维已移除，保留原特征
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(input_file):
    """
    加载提取的特征数据
    
    Args:
        input_file: 输入的CSV文件路径
        
    Returns:
        DataFrame: 加载的数据
    """
    import os
    
    print(f"正在加载数据: {input_file}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")
    
    # 获取文件大小
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"  文件大小: {file_size:.2f} MB")
    
    # 对于大文件（>100MB），直接使用分块读取以避免超时
    if file_size > 100:
        print("  文件较大，使用优化的分块读取策略...")
        print("  步骤1: 快速读取前几行以确定列结构...")
        
        # 尝试使用C引擎读取（快速），如果超时则使用Python引擎（稳定但慢）
        print("  尝试使用C引擎读取...")
        use_python_engine = False
        
        # 先尝试读取少量行来确定列结构
        try:
            print("    步骤1a: 读取前10行以确定列结构...")
            # 使用更小的样本和更短的超时
            sample = pd.read_csv(input_file, nrows=10, low_memory=False, engine='c')
            print(f"  成功！列数: {len(sample.columns)}")
            
            # 为object类型列创建dtype字典
            dtype_dict = {}
            for col in sample.columns:
                if sample[col].dtype == 'object':
                    dtype_dict[col] = 'object'
            
            print(f"  步骤2: 使用C引擎分块读取（快速模式）...")
            chunk_list = []
            chunk_size = 10000  # 使用更大的chunk size以减少迭代次数
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
                    print(f"    已读取 {total_rows:,} 行...")
            
            print(f"  步骤3: 合并所有分块...")
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"  数据形状: {df.shape}")
            print(f"  列数: {len(df.columns)}")
            print(f"  内存使用: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            return df
            
        except (TimeoutError, Exception) as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                print(f"  C引擎读取超时: {e}")
            else:
                print(f"  C引擎读取失败: {e}")
            print("  切换到Python引擎（较慢但更稳定，不会超时）...")
            use_python_engine = True
        
        # 如果C引擎失败，使用Python引擎
        if use_python_engine:
            print("  使用Python引擎分块读取（稳定模式）...")
            try:
                # 先读取前几行确定列结构，使用更稳健的方法
                print("    尝试读取样本行...")
                try:
                    sample = pd.read_csv(input_file, nrows=5, engine='python', low_memory=False)
                    print(f"  成功读取样本，列数: {len(sample.columns)}")
                except pd.errors.EmptyDataError:
                    # 如果读取样本失败，尝试直接读取整个文件
                    print("  样本读取失败，尝试直接读取整个文件...")
                    df = pd.read_csv(input_file, engine='python', low_memory=False)
                    print(f"  成功！数据形状: {df.shape}")
                    return df
                
                chunk_list = []
                chunk_size = 10000  # 使用更大的chunk size
                total_rows = 0
                
                # 使用try-except处理可能的空chunk
                print("    开始分块读取...")
                reader = pd.read_csv(
                    input_file,
                    chunksize=chunk_size,
                    engine='python',
                    low_memory=False
                )
                
                chunk_count = 0
                for i, chunk in enumerate(reader):
                    if len(chunk) > 0:  # 只处理非空chunk
                        chunk_list.append(chunk)
                        total_rows += len(chunk)
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(f"    已读取 {total_rows:,} 行...")
                    else:
                        print(f"    警告: 第 {i+1} 个chunk为空，跳过")
                        break  # 如果遇到空chunk，可能已经读完
                
                if len(chunk_list) == 0:
                    raise ValueError("没有读取到任何数据")
                
                print(f"  合并所有分块...")
                df = pd.concat(chunk_list, ignore_index=True)
                print(f"  数据形状: {df.shape}")
                print(f"  列数: {len(df.columns)}")
                return df
            except pd.errors.EmptyDataError as e:
                print(f"  错误: 文件为空或无法解析: {e}")
                # 尝试直接读取整个文件
                print("  尝试直接读取整个文件...")
                try:
                    df = pd.read_csv(input_file, engine='python', low_memory=False)
                    print(f"  成功！数据形状: {df.shape}")
                    return df
                except Exception as e2:
                    raise ValueError(f"无法读取文件: {e2}")
            except Exception as e:
                print(f"  Python引擎读取失败: {e}")
                # 最后尝试：使用更保守的设置
                print("  尝试使用更保守的设置读取...")
                try:
                    # 尝试不同的参数组合
                    read_kwargs = {
                        'engine': 'python',
                        'low_memory': False,
                        'sep': ','
                    }
                    # 对于pandas >= 1.3.0，使用on_bad_lines
                    try:
                        df = pd.read_csv(input_file, **read_kwargs, on_bad_lines='skip')
                    except TypeError:
                        # 对于旧版本pandas，使用error_bad_lines=False
                        try:
                            df = pd.read_csv(input_file, **read_kwargs, error_bad_lines=False)
                        except TypeError:
                            # 如果都不支持，直接读取
                            df = pd.read_csv(input_file, **read_kwargs)
                    print(f"  成功！数据形状: {df.shape}")
                    return df
                except Exception as e2:
                    raise ValueError(f"所有读取方法都失败: {e2}")
    else:
        # 对于小文件，直接读取
        print("  开始读取CSV文件...")
        try:
            df = pd.read_csv(input_file, low_memory=False)
            print(f"  数据形状: {df.shape}")
            print(f"  列数: {len(df.columns)}")
            print(f"  内存使用: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            return df
        except Exception as e:
            print(f"  直接读取失败: {e}")
            print("  尝试使用分块读取...")
            # 如果直接读取失败，尝试分块读取
            chunk_list = []
            chunk_size = 10000
            for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)):
                chunk_list.append(chunk)
                if (i + 1) % 10 == 0:
                    print(f"    已读取 {((i + 1) * chunk_size):,} 行...")
            df = pd.concat(chunk_list, ignore_index=True)
            print(f"  数据形状: {df.shape}")
            print(f"  列数: {len(df.columns)}")
            return df


def analyze_data(df):
    """
    分析数据的基本统计信息
    
    Args:
        df: 输入DataFrame
    """
    print("\n" + "="*60)
    print("数据基本信息分析")
    print("="*60)
    
    print(f"\n数据形状: {df.shape}")
    print(f"  行数: {df.shape[0]}")
    print(f"  列数: {df.shape[1]}")
    
    # 数据类型统计
    print("\n数据类型统计:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_counts = df.isnull().sum()
    missing_ratio = (missing_counts / len(df)) * 100
    missing_info = pd.DataFrame({
        '缺失数量': missing_counts,
        '缺失比例(%)': missing_ratio
    })
    missing_info = missing_info[missing_info['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
    
    if len(missing_info) > 0:
        print(f"  有缺失值的列数: {len(missing_info)}")
        print(f"  缺失值总数: {missing_counts.sum()}")
        print(f"  缺失值总比例: {(missing_counts.sum() / (len(df) * len(df.columns))) * 100:.2f}%")
        print("\n  缺失值最多的前10列:")
        for idx, row in missing_info.head(10).iterrows():
            print(f"    {idx}: {row['缺失数量']} ({row['缺失比例(%)']:.2f}%)")
    else:
        print("  无缺失值")
    
    # 数值特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        print(f"\n数值特征统计 (共{len(numeric_cols)}列):")
        print(f"  均值范围: [{df[numeric_cols].mean().min():.2f}, {df[numeric_cols].mean().max():.2f}]")
        print(f"  标准差范围: [{df[numeric_cols].std().min():.2f}, {df[numeric_cols].std().max():.2f}]")
        
        # 检查异常值（使用IQR方法）
        print("\n  异常值检测 (使用IQR方法):")
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
            print(f"    有异常值的列数: {len(outlier_counts)}")
            sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"    异常值最多的前5列:")
            for col, count in sorted_outliers[:5]:
                print(f"      {col}: {count} 个异常值")
        else:
            print("    未检测到明显异常值")
    
    # 分类特征统计
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(categorical_cols) > 0:
        print(f"\n分类特征统计 (共{len(categorical_cols)}列):")
        unique_counts = df[categorical_cols].nunique()
        print(f"  唯一值数量范围: [{unique_counts.min()}, {unique_counts.max()}]")
        print(f"  唯一值最多的前5列:")
        for col in unique_counts.nlargest(5).index:
            print(f"    {col}: {unique_counts[col]} 个唯一值")


def remove_duplicates(df):
    """
    移除重复行
    
    Args:
        df: 输入DataFrame
        
    Returns:
        DataFrame: 去重后的DataFrame
    """
    print("\n" + "="*60)
    print("移除重复值")
    print("="*60)
    
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    removed_rows = initial_rows - len(df_cleaned)
    
    print(f"  原始行数: {initial_rows}")
    print(f"  去重后行数: {len(df_cleaned)}")
    print(f"  移除重复行数: {removed_rows}")
    
    return df_cleaned


def handle_missing_values(df, strategy='median'):
    """
    处理缺失值
    
    Args:
        df: 输入DataFrame
        strategy: 填充策略 ('median', 'mean', 'mode', 'drop')
        
    Returns:
        DataFrame: 处理后的DataFrame
    """
    print("\n" + "="*60)
    print(f"处理缺失值 (策略: {strategy})")
    print("="*60)
    
    df_cleaned = df.copy()
    
    # 分离数值列和非数值列
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"  数值列数: {len(numeric_cols)}")
    print(f"  分类列数: {len(categorical_cols)}")
    
    if strategy == 'drop':
        # 删除包含缺失值的行
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        removed_rows = initial_rows - len(df_cleaned)
        print(f"  删除行数: {removed_rows}")
    else:
        # 对数值列使用指定策略填充
        if len(numeric_cols) > 0:
            # 确保只处理实际存在的列
            existing_numeric_cols = [col for col in numeric_cols if col in df_cleaned.columns]
            if len(existing_numeric_cols) > 0:
                # 分离全部缺失的列和部分缺失的列
                all_missing_cols = []
                partial_missing_cols = []
                
                for col in existing_numeric_cols:
                    missing_count = df_cleaned[col].isnull().sum()
                    if missing_count == len(df_cleaned):
                        # 全部缺失，用0填充
                        all_missing_cols.append(col)
                    elif missing_count > 0:
                        # 部分缺失，用imputer填充
                        partial_missing_cols.append(col)
                    # 没有缺失的列不需要处理
                
                # 处理全部缺失的列
                if len(all_missing_cols) > 0:
                    df_cleaned[all_missing_cols] = 0
                    print(f"  全部缺失的列用0填充: {len(all_missing_cols)} 列")
                
                # 处理部分缺失的列
                if len(partial_missing_cols) > 0:
                    if strategy == 'median':
                        imputer = SimpleImputer(strategy='median')
                    elif strategy == 'mean':
                        imputer = SimpleImputer(strategy='mean')
                    else:
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    
                    # fit_transform 返回 numpy 数组，需要转换回 DataFrame
                    imputed_data = imputer.fit_transform(df_cleaned[partial_missing_cols])
                    df_cleaned[partial_missing_cols] = pd.DataFrame(
                        imputed_data, 
                        columns=partial_missing_cols, 
                        index=df_cleaned.index
                    )
                    print(f"  部分缺失的列填充完成 (策略: {strategy}): {len(partial_missing_cols)} 列")
                
                print(f"  数值列填充完成，共处理 {len(existing_numeric_cols)} 列")
            else:
                print(f"  警告: 未找到有效的数值列")
        
        # 对分类列使用众数或0填充
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(0)
            print(f"  分类列填充完成")
    
    # 验证
    remaining_missing = df_cleaned.isnull().sum().sum()
    print(f"  剩余缺失值: {remaining_missing}")
    
    return df_cleaned


def handle_outliers(df, method='clip'):
    """
    处理异常值
    
    Args:
        df: 输入DataFrame
        method: 处理方法 ('clip', 'remove', 'none')
        
    Returns:
        DataFrame: 处理后的DataFrame
    """
    print("\n" + "="*60)
    print(f"处理异常值 (方法: {method})")
    print("="*60)
    
    if method == 'none':
        print("  跳过异常值处理")
        return df
    
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'clip':
        # 使用IQR方法裁剪异常值
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
        
        print(f"  裁剪的异常值总数: {clipped_count}")
    
    elif method == 'remove':
        # 删除包含异常值的行
        initial_rows = len(df_cleaned)
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        removed_rows = initial_rows - len(df_cleaned)
        print(f"  删除的行数: {removed_rows}")
    
    return df_cleaned


def remove_low_variance_features(df, threshold=0.01):
    """
    移除低方差特征
    
    Args:
        df: 输入DataFrame
        threshold: 方差阈值
        
    Returns:
        DataFrame: 处理后的DataFrame
        list: 被移除的列名列表
    """
    print("\n" + "="*60)
    print(f"移除低方差特征 (阈值: {threshold})")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("  无数值特征，跳过")
        return df, []
    
    # 计算方差
    variances = df[numeric_cols].var()
    low_variance_cols = variances[variances < threshold].index.tolist()
    
    if len(low_variance_cols) > 0:
        print(f"  移除 {len(low_variance_cols)} 个低方差特征")
        df_cleaned = df.drop(columns=low_variance_cols)
        return df_cleaned, low_variance_cols
    else:
        print(f"  未发现低方差特征")
        return df, []


def remove_highly_correlated_features(df, threshold=0.95):
    """
    移除高度相关的特征
    
    Args:
        df: 输入DataFrame
        threshold: 相关系数阈值
        
    Returns:
        DataFrame: 处理后的DataFrame
        list: 被移除的列名列表
    """
    print("\n" + "="*60)
    print(f"移除高度相关特征 (阈值: {threshold})")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("  数值特征不足，跳过")
        return df, []
    
    # 计算相关系数矩阵
    corr_matrix = df[numeric_cols].corr().abs()
    
    # 找到高度相关的特征对
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 找出需要移除的列
    to_remove = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > threshold)]
    
    if len(to_remove) > 0:
        print(f"  移除 {len(to_remove)} 个高度相关特征")
        df_cleaned = df.drop(columns=to_remove)
        return df_cleaned, to_remove
    else:
        print(f"  未发现高度相关特征")
        return df, []


def engineer_features(df):
    """
    特征工程：创建关键比率特征（与 xgboost_baseline.py 一致）
    这些特征对社交网络预测非常重要
    
    Args:
        df: 输入DataFrame
        
    Returns:
        DataFrame: 添加了工程特征后的DataFrame
    """
    print("\n" + "="*60)
    print("特征工程：创建比率特征")
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
        # 只对非缺失值计算比率
        mask = df_engineered[followers_col].notna() & df_engineered[friends_col].notna()
        df_engineered['follower_following_ratio'] = 0.0
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            df_engineered.loc[mask, followers_col] / 
            (df_engineered.loc[mask, friends_col] + 1)
        )
        print(f"  ✓ 创建: follower_following_ratio (使用 {followers_col} / {friends_col})")
        print(f"    非空比率值: {mask.sum()}/{len(df_engineered)}")
    
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
        
        # 计算账户年龄
        def calculate_account_age(created_at_str):
            if pd.isna(created_at_str):
                return None
            try:
                # Twitter created_at格式: "Wed Mar 17 12:59:53 +0000 2021"
                created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
                now = datetime.now(created_at.tzinfo)
                return (now - created_at).days
            except:
                return None
        
        account_ages = df_engineered[created_col].apply(calculate_account_age)
        df_engineered['account_age_days'] = account_ages
        
        # 计算活动率（只对有效值计算）
        mask = df_engineered[statuses_col].notna() & df_engineered['account_age_days'].notna()
        df_engineered['user_activity_rate'] = 0.0
        df_engineered.loc[mask, 'user_activity_rate'] = (
            df_engineered.loc[mask, statuses_col] / 
            (df_engineered.loc[mask, 'account_age_days'] + 1)
        )
        print(f"  ✓ 创建: user_activity_rate (使用 {statuses_col} / account_age_days)")
        print(f"    非空活动率值: {mask.sum()}/{len(df_engineered)}")
    
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
        
        # 只对有效值计算参与率
        mask = (df_engineered[retweet_col].notna() & 
                df_engineered[favorite_col].notna() & 
                df_engineered[followers_col].notna())
        df_engineered['tweet_engagement_rate'] = 0.0
        df_engineered.loc[mask, 'tweet_engagement_rate'] = (
            (df_engineered.loc[mask, retweet_col] + 
             df_engineered.loc[mask, favorite_col]) / 
            (df_engineered.loc[mask, followers_col] + 1)
        )
        print(f"  ✓ 创建: tweet_engagement_rate")
        print(f"    非空参与率值: {mask.sum()}/{len(df_engineered)}")
    
    print(f"  特征工程完成，新增特征数: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered


def encode_categorical_features(df):
    """
    编码分类特征为数值
    
    Args:
        df: 输入DataFrame
        
    Returns:
        DataFrame: 编码后的DataFrame
    """
    print("\n" + "="*60)
    print("编码分类特征")
    print("="*60)
    
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("  无分类特征，跳过")
        return df_encoded
    
    print(f"  需要编码的分类特征数: {len(categorical_cols)}")
    
    # 对于每个分类特征，使用标签编码或独热编码
    # 如果唯一值数量少（<10），使用独热编码；否则使用标签编码
    for col in categorical_cols:
        unique_count = df_encoded[col].nunique()
        
        if unique_count < 10 and unique_count > 1:
            # 使用独热编码
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            print(f"    {col}: 独热编码 ({unique_count} 个唯一值)")
        else:
            # 使用标签编码（转换为数值）
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            print(f"    {col}: 标签编码 ({unique_count} 个唯一值)")
    
    print(f"  编码完成，当前列数: {len(df_encoded.columns)}")
    
    return df_encoded


def normalize_features(df, method='standard'):
    """
    标准化/归一化特征
    
    Args:
        df: 输入DataFrame
        method: 标准化方法 ('standard', 'robust', 'none')
        
    Returns:
        DataFrame: 标准化后的DataFrame
        scaler: 使用的scaler对象（用于后续数据转换）
    """
    print("\n" + "="*60)
    print(f"特征标准化 (方法: {method})")
    print("="*60)
    
    if method == 'none':
        print("  跳过标准化")
        return df, None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("  无数值特征，跳过")
        return df, None
    
    df_normalized = df.copy()
    
    # 确保只处理实际存在的列
    existing_numeric_cols = [col for col in numeric_cols if col in df_normalized.columns]
    
    if len(existing_numeric_cols) == 0:
        print("  未找到有效的数值特征，跳过标准化")
        return df, None
    
    if method == 'standard':
        scaler = StandardScaler()
        print(f"  使用StandardScaler标准化 {len(existing_numeric_cols)} 个数值特征")
    elif method == 'robust':
        scaler = RobustScaler()
        print(f"  使用RobustScaler标准化 {len(existing_numeric_cols)} 个数值特征")
    else:
        print(f"  未知方法，跳过标准化")
        return df, None
    
    # fit_transform 返回 numpy 数组，需要转换回 DataFrame
    normalized_data = scaler.fit_transform(df_normalized[existing_numeric_cols])
    df_normalized[existing_numeric_cols] = pd.DataFrame(
        normalized_data,
        columns=existing_numeric_cols,
        index=df_normalized.index
    )
    print(f"  标准化完成")
    
    return df_normalized, scaler


def reduce_dimensionality(df, method='none', n_components=None, variance_ratio=0.95):
    """
    降维处理（已禁用，保留原特征）
    
    Args:
        df: 输入DataFrame
        method: 降维方法（已禁用，始终返回原数据）
        n_components: 主成分数量（已禁用）
        variance_ratio: 保留的方差比例（已禁用）
        
    Returns:
        DataFrame: 原始DataFrame（不进行降维）
        transformer: None（降维已禁用）
    """
    print("\n" + "="*60)
    print(f"降维处理 (方法: {method})")
    print("="*60)
    
    print("  降维功能已禁用，保留所有原始特征")
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
    print("数据清洗流程（保留原特征）")
    print("="*60)
    
    transformers = {}
    
    # ========== 训练集处理 ==========
    print("\n[处理训练集]")
    # 1. 加载数据
    df_train = load_data(input_file)
    
    # ========== 如果提供了测试集，先对齐特征列 ==========
    if test_input_file and test_output_file:
        import os
        if os.path.exists(test_input_file):
            print("\n[预对齐特征列]")
            print("  正在加载测试集以对齐特征列...")
            df_test_preview = load_data(test_input_file)
            
            # 获取训练集和测试集的所有列
            train_cols = set(df_train.columns)
            test_cols = set(df_test_preview.columns)
            
            # 找出差异
            missing_in_test = train_cols - test_cols
            missing_in_train = test_cols - train_cols
            common_cols = train_cols & test_cols
            
            print(f"  训练集列数: {len(train_cols)}")
            print(f"  测试集列数: {len(test_cols)}")
            print(f"  共同列数: {len(common_cols)}")
            
            if len(missing_in_test) > 0:
                print(f"  测试集缺失的列（将在测试集中添加，用NaN填充）: {len(missing_in_test)}")
            if len(missing_in_train) > 0:
                print(f"  训练集缺失的列（将在训练集中添加，用NaN填充）: {len(missing_in_train)}")
            
            # 确定统一的列列表（按字母顺序排序以确保一致性）
            unified_cols = sorted(list(train_cols | test_cols))
            
            # 对齐训练集
            for col in unified_cols:
                if col not in df_train.columns:
                    df_train[col] = None
            df_train = df_train[unified_cols]
            
            print(f"  ✓ 训练集已对齐为 {len(unified_cols)} 列")
            
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
            print(f"  已移除低方差列: {len(removed_low_var_cols)} 个")
    
    # 8. 移除高度相关特征（fit on train）
    removed_high_corr_cols = []
    if remove_high_corr:
        df_train, removed_high_corr_cols = remove_highly_correlated_features(df_train, threshold=0.95)
        if len(removed_high_corr_cols) > 0:
            print(f"  已移除高度相关列: {len(removed_high_corr_cols)} 个")
    
    # 保存要移除的列名
    transformers['removed_cols'] = removed_low_var_cols + removed_high_corr_cols
    
    # 9. 标准化特征（fit on train）
    # 保存标准化前的列名（用于测试集标准化前的对齐）
    transformers['train_columns_pre_normalize'] = df_train.columns.tolist()
    
    # 保存标准化前使用的数值列名
    numeric_cols_before_normalize = df_train.select_dtypes(include=[np.number]).columns.tolist()
    transformers['train_numeric_cols_for_scaler'] = numeric_cols_before_normalize
    
    df_train, scaler = normalize_features(df_train, method=normalize_method)
    transformers['scaler'] = scaler
    transformers['scaler_method'] = normalize_method
    
    # 10. 降维（已禁用，保留原特征）
    # PCA降维已移除，保留所有原始特征
    if reduce_method != 'none':
        print("\n警告: reduce_method不为'none'，但PCA降维功能已移除，将保留所有特征")
    
    # 保存训练集的最终列顺序（用于测试集最终对齐）
    transformers['train_columns'] = df_train.columns.tolist()
    
    # 11. 最终统计
    print("\n" + "="*60)
    print("训练集最终数据统计")
    print("="*60)
    print(f"  最终数据形状: {df_train.shape}")
    print(f"  最终列数: {len(df_train.columns)}")
    print(f"  缺失值总数: {df_train.isnull().sum().sum()}")
    
    # 12. 保存训练集结果
    print(f"\n正在保存训练集到: {output_file}")
    df_train.to_csv(output_file, index=False)
    print(f"✓ 成功保存清洗后的训练集数据到 {output_file}")
    
    # ========== 测试集处理 ==========
    df_test = None
    if test_input_file and test_output_file:
        # 检查测试集输入文件是否存在
        import os
        if not os.path.exists(test_input_file):
            print(f"\n警告: 测试集输入文件不存在: {test_input_file}")
            print("跳过测试集处理。如果只想处理训练集，这是正常的。")
            print("如果希望处理测试集，请确保文件存在或使用 --test-input 指定正确的路径。")
        else:
            print("\n" + "="*60)
            print("[处理测试集]")
            print("="*60)
            
            # 1. 加载测试集
            df_test = load_data(test_input_file)
            
            # 1.5. 对齐特征列（使用训练集的统一列列表）
            unified_cols = transformers.get('unified_columns')
            if unified_cols:
                print("\n  对齐测试集特征列...")
                for col in unified_cols:
                    if col not in df_test.columns:
                        df_test[col] = None
                df_test = df_test[unified_cols]
                print(f"  ✓ 测试集已对齐为 {len(unified_cols)} 列")
            
            # 2. 跳过移除重复值（测试集必须保持原始行数，以便与ID对应）
            print("\n  跳过移除重复值（测试集保持原始行数以匹配ID）")
            initial_test_rows = len(df_test)
            
            # 2.5. 特征工程：创建比率特征（在处理缺失值之前，需要原始数据）
            # 这必须在编码分类特征之前，因为需要原始的日期字符串和数值
            df_test = engineer_features(df_test)
            
            # 2.6. 对齐特征工程后的列（确保测试集有训练集的所有工程特征）
            train_cols_after_fe = transformers.get('train_columns_after_feature_eng', [])
            if train_cols_after_fe:
                missing_cols = set(train_cols_after_fe) - set(df_test.columns)
                extra_cols = set(df_test.columns) - set(train_cols_after_fe)
                
                # 添加缺失的列（用0填充）
                if len(missing_cols) > 0:
                    for col in missing_cols:
                        df_test[col] = 0
                    print(f"  特征工程后对齐：添加了 {len(missing_cols)} 个缺失列")
                
                # 移除多余的列
                if len(extra_cols) > 0:
                    df_test = df_test.drop(columns=list(extra_cols))
                    print(f"  特征工程后对齐：移除了 {len(extra_cols)} 个多余列")
                
                # 确保列顺序一致
                df_test = df_test[train_cols_after_fe]
                print(f"  ✓ 测试集特征工程后对齐为 {len(train_cols_after_fe)} 列")
            
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
                            
                            # 只保留训练集中存在的列
                            for onehot_col in train_onehot_cols:
                                if onehot_col in dummies.columns:
                                    df_test[onehot_col] = dummies[onehot_col]
                            
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
                print(f"  标准化前对齐：添加了 {len(missing_cols)} 个缺失列")
            
            # 移除多余的列（特别是独热编码产生的额外列）
            if len(extra_cols) > 0:
                df_test = df_test.drop(columns=list(extra_cols))
                print(f"  标准化前对齐：移除了 {len(extra_cols)} 个多余列")
            
            # 确保列顺序一致
            df_test = df_test[train_columns_pre_normalize]
            
            # 8. 标准化（使用训练集的scaler）
            scaler = transformers.get('scaler')
            scaler_method = transformers.get('scaler_method', 'none')
            if scaler is not None and scaler_method != 'none':
                # 获取训练集标准化时使用的列名（使用scaler的实际特征名）
                if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
                    # 使用scaler实际拟合时的特征名（这是最准确的）
                    train_scaler_numeric_cols = list(scaler.feature_names_in_)
                else:
                    # 如果没有feature_names_in_，使用保存的数值列名
                    train_scaler_numeric_cols = transformers.get('train_numeric_cols_for_scaler', [])
                    if not train_scaler_numeric_cols:
                        # 最后的备选方案：从保存的列名中推断
                        train_columns_final = transformers.get('train_columns', [])
                        train_scaler_numeric_cols = train_columns_final
                
                # 确保测试集有训练集标准化时需要的所有列
                missing_cols = set(train_scaler_numeric_cols) - set(df_test.columns)
                extra_cols = set(df_test.columns) - set(train_scaler_numeric_cols)
                
                # 添加缺失的列（用0填充）
                if len(missing_cols) > 0:
                    for col in missing_cols:
                        df_test[col] = 0
                    print(f"  标准化前：添加了 {len(missing_cols)} 个缺失列")
                
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
                    print(f"  最终对齐：添加了 {len(missing_cols)} 个缺失列")
                
                # 移除多余的列
                if len(extra_cols) > 0:
                    df_test = df_test.drop(columns=list(extra_cols))
                    print(f"  最终对齐：移除了 {len(extra_cols)} 个多余列")
                
                # 确保列顺序一致
                df_test = df_test[train_columns]
                print(f"  ✓ 测试集最终对齐为 {len(train_columns)} 列")
            
            # 10. 最终统计
            print("\n" + "="*60)
            print("测试集最终数据统计")
            print("="*60)
            print(f"  最终数据形状: {df_test.shape}")
            print(f"  最终列数: {len(df_test.columns)}")
            print(f"  缺失值总数: {df_test.isnull().sum().sum()}")
            
            # 11. 保存测试集结果
            print(f"\n正在保存测试集到: {test_output_file}")
            df_test.to_csv(test_output_file, index=False)
            print(f"✓ 成功保存清洗后的测试集数据到 {test_output_file}")
    
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
        parser = argparse.ArgumentParser(description='数据清洗和降维脚本')
        parser.add_argument('--input', type=str, default='cache/extracted_features.csv',
                            help='训练集输入CSV文件路径')
        parser.add_argument('--output', type=str, default='cache/cleaned_features.csv',
                            help='训练集输出CSV文件路径')
        parser.add_argument('--test-input', type=str, default='cache/test_extracted_features.csv',
                            help='测试集输入CSV文件路径')
        parser.add_argument('--test-output', type=str, default='cache/test_cleaned_features.csv',
                            help='测试集输出CSV文件路径')
        parser.add_argument('--missing-strategy', type=str, default='median',
                            choices=['median', 'mean', 'mode', 'drop'],
                            help='缺失值处理策略')
        parser.add_argument('--outlier-method', type=str, default='clip',
                            choices=['clip', 'remove', 'none'],
                            help='异常值处理方法')
        parser.add_argument('--remove-low-var', action='store_true', default=False,
                            help='移除低方差特征（默认：不移除，适合XGBoost）')
        parser.add_argument('--remove-high-corr', action='store_true', default=False,
                            help='移除高度相关特征（默认：不移除，适合XGBoost）')
        parser.add_argument('--normalize', type=str, default='none',
                            choices=['standard', 'robust', 'none'],
                            help='标准化方法（默认：none，XGBoost树模型不需要标准化）')
        parser.add_argument('--reduce', type=str, default='none',
                            choices=['pca', 'none'],
                            help='降维方法（已禁用，保留原特征）')
        parser.add_argument('--variance-ratio', type=float, default=0.95,
                            help='PCA保留的方差比例（已禁用）')
        
        args = parser.parse_args()
        
        print(f"项目根目录: {project_root}")
        
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


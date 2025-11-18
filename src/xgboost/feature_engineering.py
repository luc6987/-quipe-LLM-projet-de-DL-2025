"""
特征工程模块：实现指南第一部分的所有特征工程逻辑 / Feature engineering module: Implements all feature engineering logic from guide part 1
根据竞赛定义，创建强大的非文本特征集 / Create powerful non-text feature set according to competition definition
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')


def calculate_account_age(created_at_str):
    """
    计算账户年龄（天数） / Calculate account age (in days)
    
    Args:
        created_at_str: Twitter created_at格式的字符串，例如 "Wed Mar 17 12:59:53 +0000 2021" / Twitter created_at format string, e.g. "Wed Mar 17 12:59:53 +0000 2021"
        
    Returns:
        int: 账户年龄（天数），如果解析失败返回None / Account age (days), returns None if parsing fails
    """
    if pd.isna(created_at_str):
        return None
    
    try:
        # Twitter created_at格式: "Wed Mar 17 12:59:53 +0000 2021"
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        now = datetime.now(created_at.tzinfo)
        age_days = (now - created_at).days
        return age_days
    except (ValueError, TypeError):
        return None


def engineer_features(df, is_train=True):
    """
    特征工程：创建所有指南中提到的关键特征 / Feature engineering: Create all key features mentioned in guide
    
    根据指南，特征分为四类： / According to guide, features are divided into four categories:
    1. 黄金特征：follower_following_ratio, follower_following_diff / Golden features: follower_following_ratio, follower_following_diff
    2. 用户级元数据：account_age_days, verified, default_profile_image, listed_count等 / User-level metadata: account_age_days, verified, default_profile_image, listed_count, etc.
    3. 推文级元数据：is_reply, is_quote, retweet_count, favorite_count等 / Tweet-level metadata: is_reply, is_quote, retweet_count, favorite_count, etc.
    4. Entities特征：hashtags_count, urls_count, mentions_count / Entities features: hashtags_count, urls_count, mentions_count
    
    Args:
        df: 输入DataFrame（已经过json_normalize处理） / Input DataFrame (already processed by json_normalize)
        is_train: 是否为训练集（用于决定是否处理标签） / Whether it's training set (to decide whether to process labels)
        
    Returns:
        DataFrame: 添加了工程特征后的DataFrame / DataFrame with engineered features added
        dict: 包含编码器等需要保存的对象 / Dictionary containing encoders and other objects to save
    """
    print("\n" + "="*60)
    print("Feature Engineering: Creating Key Features")
    print("="*60)
    
    df_engineered = df.copy()
    transformers = {}
    
    # ========== 1. 黄金特征：follower_following_ratio ========== / 1. Golden features: follower_following_ratio
    # 这是竞赛定义的核心特征：Influencer = 高关注者/关注中比例 / This is the core feature defined by competition: Influencer = high follower/following ratio
    print("\n[1] Creating Golden Features...")
    
    # 优先使用主推文的用户特征，而不是quoted_status的用户特征 / Prefer main tweet user features, not quoted_status user features
    user_followers_cols = [col for col in df.columns 
                          if 'followers_count' in col.lower() 
                          and 'quoted_status' not in col.lower()]
    if not user_followers_cols:
        user_followers_cols = [col for col in df.columns if 'followers_count' in col.lower()]
    
    user_friends_cols = [col for col in df.columns 
                        if 'friends_count' in col.lower() 
                        and 'quoted_status' not in col.lower()]
    if not user_friends_cols:
        user_friends_cols = [col for col in df.columns if 'friends_count' in col.lower()]
    
    if user_followers_cols and user_friends_cols:
        followers_col = user_followers_cols[0]
        friends_col = user_friends_cols[0]
        
        # follower_following_ratio: 核心预测器 / Core predictor
        mask = df_engineered[followers_col].notna() & df_engineered[friends_col].notna()
        df_engineered['follower_following_ratio'] = np.nan
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            df_engineered.loc[mask, followers_col] / 
            (df_engineered.loc[mask, friends_col] + 1e-6)  # epsilon防止除零 / epsilon to prevent division by zero
        )
        print(f"  ✓ follower_following_ratio (using {followers_col} / {friends_col})")
        print(f"    Non-null values: {mask.sum()}/{len(df_engineered)}")
        
        # follower_following_diff: 绝对差异特征 / Absolute difference feature
        df_engineered['follower_following_diff'] = np.nan
        df_engineered.loc[mask, 'follower_following_diff'] = (
            df_engineered.loc[mask, followers_col] - 
            df_engineered.loc[mask, friends_col]
        )
        print(f"  ✓ follower_following_diff")
    else:
        print("  ⚠ Warning: followers_count or friends_count columns not found")
    
    # ========== 2. 用户级元数据特征 ========== / 2. User-level metadata features
    print("\n[2] Creating User-level Metadata Features...")
    
    # 2.1 account_age_days: 账户年龄 / Account age
    user_created_cols = [col for col in df.columns 
                        if 'user.created_at' in col.lower() 
                        and 'quoted_status' not in col.lower()]
    if not user_created_cols:
        user_created_cols = [col for col in df.columns 
                            if 'created_at' in col.lower() 
                            and 'user' in col.lower()]
    
    if user_created_cols:
        created_col = user_created_cols[0]
        account_ages = df_engineered[created_col].apply(calculate_account_age)
        df_engineered['account_age_days'] = account_ages
        print(f"  ✓ account_age_days (using {created_col})")
        print(f"    Non-null values: {account_ages.notna().sum()}/{len(df_engineered)}")
    else:
        df_engineered['account_age_days'] = np.nan
        print("  ⚠ Warning: user.created_at column not found")
    
    # 2.2 user.verified: 验证状态（强影响力信号） / Verification status (strong influence signal)
    user_verified_cols = [col for col in df.columns 
                         if 'user.verified' in col.lower() 
                         and 'quoted_status' not in col.lower()]
    if user_verified_cols:
        verified_col = user_verified_cols[0]
        # 转换为整数：True -> 1, False -> 0, None -> np.nan / Convert to integer: True -> 1, False -> 0, None -> np.nan
        df_engineered['user_verified'] = df_engineered[verified_col].apply(
            lambda x: 1 if x is True else (0 if x is False else np.nan)
        )
        print(f"  ✓ user_verified")
    else:
        df_engineered['user_verified'] = np.nan
        print("  ⚠ Warning: user.verified column not found")
    
    # 2.3 user.default_profile_image: 默认头像（低活跃度信号）
    user_default_img_cols = [col for col in df.columns 
                            if 'user.default_profile_image' in col.lower() 
                            and 'quoted_status' not in col.lower()]
    if user_default_img_cols:
        default_img_col = user_default_img_cols[0]
        df_engineered['user_default_profile_image'] = df_engineered[default_img_col].apply(
            lambda x: 1 if x is True else (0 if x is False else np.nan)
        )
        print(f"  ✓ user_default_profile_image")
    else:
        df_engineered['user_default_profile_image'] = np.nan
    
    # 2.4 user.listed_count: 被列表收录次数（强影响力信号）
    user_listed_cols = [col for col in df.columns 
                       if 'user.listed_count' in col.lower() 
                       and 'quoted_status' not in col.lower()]
    if user_listed_cols:
        listed_col = user_listed_cols[0]
        df_engineered['user_listed_count'] = df_engineered[listed_col]
        print(f"  ✓ user_listed_count")
    else:
        df_engineered['user_listed_count'] = np.nan
    
    # 2.5 user.statuses_count, user.favourites_count
    user_statuses_cols = [col for col in df.columns 
                         if 'user.statuses_count' in col.lower() 
                         and 'quoted_status' not in col.lower()]
    if user_statuses_cols:
        statuses_col = user_statuses_cols[0]
        df_engineered['user_statuses_count'] = df_engineered[statuses_col]
    
    user_favourites_cols = [col for col in df.columns 
                           if 'user.favourites_count' in col.lower() 
                           and 'quoted_status' not in col.lower()]
    if user_favourites_cols:
        favourites_col = user_favourites_cols[0]
        df_engineered['user_favourites_count'] = df_engineered[favourites_col]
    
    # ========== 3. 速率特征（关键的衍生特征）========== / 3. Rate features (key derived features)
    # 消除账户年龄的混淆效应，发现真实的用户行为 / Eliminate confounding effect of account age, discover true user behavior
    print("\n[3] Creating Rate Features...")
    
    if 'account_age_days' in df_engineered.columns and 'user_statuses_count' in df_engineered.columns:
        mask = df_engineered['account_age_days'].notna() & df_engineered['user_statuses_count'].notna()
        df_engineered['tweets_per_day'] = np.nan
        df_engineered.loc[mask, 'tweets_per_day'] = (
            df_engineered.loc[mask, 'user_statuses_count'] / 
            (df_engineered.loc[mask, 'account_age_days'] + 1e-6)
        )
        print(f"  ✓ tweets_per_day")
    
    if 'account_age_days' in df_engineered.columns and 'user_favourites_count' in df_engineered.columns:
        mask = df_engineered['account_age_days'].notna() & df_engineered['user_favourites_count'].notna()
        df_engineered['favorites_per_day'] = np.nan
        df_engineered.loc[mask, 'favorites_per_day'] = (
            df_engineered.loc[mask, 'user_favourites_count'] / 
            (df_engineered.loc[mask, 'account_age_days'] + 1e-6)
        )
        print(f"  ✓ favorites_per_day")
    
    if 'account_age_days' in df_engineered.columns and 'user_listed_count' in df_engineered.columns:
        mask = df_engineered['account_age_days'].notna() & df_engineered['user_listed_count'].notna()
        df_engineered['listed_per_day'] = np.nan
        df_engineered.loc[mask, 'listed_per_day'] = (
            df_engineered.loc[mask, 'user_listed_count'] / 
            (df_engineered.loc[mask, 'account_age_days'] + 1e-6)
        )
        print(f"  ✓ listed_per_day")
    
    if 'user_statuses_count' in df_engineered.columns and 'user_favourites_count' in df_engineered.columns:
        mask = df_engineered['user_statuses_count'].notna() & df_engineered['user_favourites_count'].notna()
        df_engineered['favorites_per_tweet'] = np.nan
        df_engineered.loc[mask, 'favorites_per_tweet'] = (
            df_engineered.loc[mask, 'user_favourites_count'] / 
            (df_engineered.loc[mask, 'user_statuses_count'] + 1e-6)
        )
        print(f"  ✓ favorites_per_tweet")
    
    # ========== 4. 推文级元数据特征 ========== / 4. Tweet-level metadata features
    print("\n[4] Creating Tweet-level Metadata Features...")
    
    # 4.1 is_reply: 核心Observer信号（互惠对话） / Core Observer signal (reciprocal conversation)
    if 'in_reply_to_user_id' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id'].notna().astype(int)
        # 将None转换为0（如果notna()返回False） / Convert None to 0 (if notna() returns False)
        df_engineered.loc[df_engineered['in_reply_to_user_id'].isna(), 'is_reply'] = 0
        print(f"  ✓ is_reply (core Observer predictor)")
    elif 'in_reply_to_user_id_str' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id_str'].notna().astype(int)
        df_engineered.loc[df_engineered['in_reply_to_user_id_str'].isna(), 'is_reply'] = 0
        print(f"  ✓ is_reply (using in_reply_to_user_id_str)")
    else:
        df_engineered['is_reply'] = 0
        print("  ⚠ Warning: in_reply_to_user_id column not found")
    
    # 4.2 is_quote: 引用推文 / Quote tweet
    if 'quoted_status' in df_engineered.columns:
        df_engineered['is_quote'] = df_engineered['quoted_status'].notna().astype(int)
        df_engineered.loc[df_engineered['quoted_status'].isna(), 'is_quote'] = 0
        print(f"  ✓ is_quote")
    elif 'quoted_status_id' in df_engineered.columns:
        df_engineered['is_quote'] = df_engineered['quoted_status_id'].notna().astype(int)
        df_engineered.loc[df_engineered['quoted_status_id'].isna(), 'is_quote'] = 0
        print(f"  ✓ is_quote (using quoted_status_id)")
    else:
        df_engineered['is_quote'] = 0
    
    # 4.3 retweet_count, favorite_count: 推文互动信号 / Tweet interaction signals
    retweet_cols = [col for col in df.columns 
                   if 'retweet_count' in col.lower() 
                   and 'quoted_status' not in col.lower()]
    if retweet_cols:
        retweet_col = retweet_cols[0]
        df_engineered['retweet_count'] = df_engineered[retweet_col]
        print(f"  ✓ retweet_count")
    else:
        df_engineered['retweet_count'] = np.nan
    
    favorite_cols = [col for col in df.columns 
                    if 'favorite_count' in col.lower() 
                    and 'quoted_status' not in col.lower()]
    if favorite_cols:
        favorite_col = favorite_cols[0]
        df_engineered['favorite_count'] = df_engineered[favorite_col]
        print(f"  ✓ favorite_count")
    else:
        df_engineered['favorite_count'] = np.nan
    
    # 4.4 possibly_sensitive
    if 'possibly_sensitive' in df_engineered.columns:
        df_engineered['possibly_sensitive'] = df_engineered['possibly_sensitive'].apply(
            lambda x: 1 if x is True else (0 if x is False else np.nan)
        )
        print(f"  ✓ possibly_sensitive")
    
    # ========== 5. Source编码（使用OrdinalEncoder）========== / 5. Source encoding (using OrdinalEncoder)
    print("\n[5] Encoding Source Feature (High Cardinality Categorical Feature)...")
    
    if 'source' in df_engineered.columns:
        # 使用OrdinalEncoder处理高基数分类特征（XGBoost推荐方法） / Use OrdinalEncoder for high cardinality categorical features (XGBoost recommended method)
        source_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # 准备数据：将NaN转换为字符串'unknown' / Prepare data: convert NaN to string 'unknown'
        source_data = df_engineered['source'].fillna('unknown').values.reshape(-1, 1)
        
        if is_train:
            # 训练集：fit_transform / Training set: fit_transform
            source_encoded = source_encoder.fit_transform(source_data)
            transformers['source_encoder'] = source_encoder
        else:
            # 测试集：只transform（使用已保存的encoder） / Test set: only transform (use saved encoder)
            if 'source_encoder' not in transformers:
                # 如果没有提供encoder，创建一个新的（不推荐，但为了兼容性） / If encoder not provided, create new one (not recommended, but for compatibility)
                source_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                source_encoded = source_encoder.fit_transform(source_data)
                transformers['source_encoder'] = source_encoder
            else:
                source_encoded = transformers['source_encoder'].transform(source_data)
        
        df_engineered['source_encoded'] = source_encoded.flatten()
        print(f"  ✓ source_encoded (using OrdinalEncoder)")
        print(f"    Unique values: {df_engineered['source'].nunique()}")
    else:
        df_engineered['source_encoded'] = np.nan
        print("  ⚠ Warning: source column not found")
    
    # ========== 6. Entities特征（非文本的文本信号）========== / 6. Entities features (non-text text signals)
    print("\n[6] Creating Entities Features...")
    
    # 6.1 hashtags_count
    if 'entities.hashtags' in df_engineered.columns:
        df_engineered['hashtags_count'] = df_engineered['entities.hashtags'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        print(f"  ✓ hashtags_count")
    else:
        df_engineered['hashtags_count'] = 0
    
    # 6.2 urls_count
    if 'entities.urls' in df_engineered.columns:
        df_engineered['urls_count'] = df_engineered['entities.urls'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        print(f"  ✓ urls_count")
    else:
        df_engineered['urls_count'] = 0
    
    # 6.3 mentions_count（Observer信号） / Observer signal
    if 'entities.user_mentions' in df_engineered.columns:
        df_engineered['mentions_count'] = df_engineered['entities.user_mentions'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        print(f"  ✓ mentions_count (Observer signal)")
    else:
        df_engineered['mentions_count'] = 0
    
    # ========== 7. 缺失值处理：统一转换为np.nan ========== / 7. Missing value handling: uniformly convert to np.nan
    # XGBoost可以原生处理np.nan，会自动学习处理缺失值的最佳分裂方向 / XGBoost can natively handle np.nan, automatically learns best split direction for missing values
    print("\n[7] Missing Value Handling: Uniformly Convert to np.nan (XGBoost Native Processing)...")
    
    # 将所有None、'None'、空字符串等统一转换为np.nan / Convert all None, 'None', empty strings, etc. to np.nan uniformly
    for col in df_engineered.columns:
        if df_engineered[col].dtype == 'object':
            # 对于object类型，将None、'None'、空字符串转换为np.nan / For object type, convert None, 'None', empty strings to np.nan
            df_engineered[col] = df_engineered[col].replace([None, 'None', ''], np.nan)
        elif df_engineered[col].dtype in ['int64', 'float64']:
            # 对于数值类型，确保None转换为np.nan / For numeric types, ensure None is converted to np.nan
            df_engineered[col] = df_engineered[col].replace([None], np.nan)
    
    print(f"  ✓ Missing value handling complete")
    print(f"    Total missing values: {df_engineered.isnull().sum().sum()}")
    
    print(f"\n{'='*60}")
    print(f"Feature Engineering Complete!")
    print(f"  Original features: {len(df.columns)}")
    print(f"  New features: {len(df_engineered.columns) - len(df.columns)}")
    print(f"  Total features: {len(df_engineered.columns)}")
    print(f"{'='*60}")
    
    return df_engineered, transformers


def extract_user_id(df):
    """
    提取用户ID用于GroupKFold验证 / Extract user ID for GroupKFold validation
    
    由于user.id可能不存在，我们需要创建用户标识符 / Since user.id may not exist, we need to create user identifier
    策略：使用user.screen_name + user.created_at的组合，或使用其他唯一标识符 / Strategy: Use combination of user.screen_name + user.created_at, or use other unique identifier
    
    Args:
        df: 输入DataFrame / Input DataFrame
        
    Returns:
        Series: 用户ID序列 / User ID Series
    """
    # 尝试多种方式提取用户ID / Try multiple methods to extract user ID
    user_id = None
    
    # 方法1: 直接使用user.id（如果存在） / Method 1: Directly use user.id (if exists)
    if 'user.id' in df.columns:
        user_id = df['user.id'].copy()
        print(f"  Using user.id as user identifier")
    elif 'user.id_str' in df.columns:
        user_id = df['user.id_str'].copy()
        print(f"  Using user.id_str as user identifier")
    # 方法2: 使用user.screen_name + user.created_at组合 / Method 2: Use combination of user.screen_name + user.created_at
    elif 'user.screen_name' in df.columns and 'user.created_at' in df.columns:
        user_id = (df['user.screen_name'].astype(str) + '_' + 
                  df['user.created_at'].astype(str))
        print(f"  Using user.screen_name + user.created_at as user identifier")
    # 方法3: 如果都没有，使用索引（不推荐，但作为最后手段） / Method 3: If none exist, use index (not recommended, but as last resort)
    else:
        user_id = pd.Series(range(len(df)), index=df.index)
        print(f"  ⚠ Warning: User identifier not found, using index (may cause data leakage)")
    
    return user_id


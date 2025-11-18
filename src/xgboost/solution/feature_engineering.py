"""
Feature Engineering Module
特征工程模块 / Module d'ingénierie des caractéristiques / Feature Engineering Module / Feature-Engineering-Modul

Implements Phase 1 of the technical blueprint: Metadata feature extraction.
实现技术蓝图第一阶段：元数据特征提取。

Key Features:
1. Network Asymmetry (网络不对称性 / Asymétrie du réseau / Network Asymmetry / Netzwerk-Asymmetrie)
2. Behavioral Rates (行为速率 / Taux comportemental / Behavioral Rates / Verhaltensraten)
3. Tweet-level Interaction Features
4. Categorical Encoding (OrdinalEncoder for source field)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import OrdinalEncoder
import logging

logger = logging.getLogger(__name__)


def calculate_account_age(created_at_str: str) -> Optional[int]:
    """
    Calculate account age in days from Twitter created_at string.
    从Twitter created_at字符串计算账户年龄（天数）。
    
    Twitter format: "Wed Mar 17 12:59:53 +0000 2021"
    
    Args:
        created_at_str: Twitter created_at format string
        
    Returns:
        int: Account age in days, or None if parsing fails
    """
    if pd.isna(created_at_str) or created_at_str is None:
        return None
    
    try:
        # Twitter created_at format: "Wed Mar 17 12:59:53 +0000 2021"
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        now = datetime.now(created_at.tzinfo)
        age_days = (now - created_at).days
        return max(0, age_days)  # Ensure non-negative
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse created_at: {created_at_str}, error: {e}")
        return None


def engineer_features(df: pd.DataFrame, is_train: bool = True, 
                     transformers: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Engineer metadata features according to the technical blueprint.
    根据技术蓝图进行元数据特征工程。
    
    Phase 1 Features:
    1. Network Asymmetry:
       - follower_following_ratio: user.followers_count / (user.friends_count + 1e-6)
       - follower_following_diff: user.followers_count - user.friends_count
    
    2. Behavioral Rates (eliminating temporal collinearity):
       - account_age_days: (current_time - user.created_at).days
       - tweets_per_day: user.statuses_count / (account_age_days + 1e-6)
       - listed_per_day: user.listed_count / (account_age_days + 1e-6)
    
    3. Tweet-level Features:
       - is_reply: 1 if in_reply_to_user_id not null, else 0
       - mentions_count, hashtags_count, urls_count: from entities lengths
    
    4. Categorical Encoding:
       - source: OrdinalEncoder (NOT One-Hot Encoding)
    
    Args:
        df: Input DataFrame (flattened JSON)
        is_train: Whether this is training data (affects transformer fitting)
        transformers: Dictionary of transformers from training set (for test set)
        
    Returns:
        Tuple of:
        - df_engineered: DataFrame with engineered features
        - transformers: Dictionary of transformers (OrdinalEncoder for source)
    """
    logger.info("="*60)
    logger.info("Feature Engineering: Creating Metadata Features")
    logger.info("="*60)
    
    df_engineered = df.copy()
    
    if transformers is None:
        transformers = {}
    
    epsilon = 1e-6  # Small value to prevent division by zero
    
    # ========== 1. Network Asymmetry Features ==========
    # 网络不对称性特征 / Caractéristiques d'asymétrie du réseau / Network Asymmetry Features
    logger.info("\n[1] Creating Network Asymmetry Features...")
    
    # Find user follower/friend columns (prefer main tweet user, not quoted_status user)
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
        
        # follower_following_ratio: Core predictor (核心预测器)
        # Formula: followers_count / (friends_count + epsilon)
        mask = df_engineered[followers_col].notna() & df_engineered[friends_col].notna()
        df_engineered['follower_following_ratio'] = np.nan
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            df_engineered.loc[mask, followers_col] / 
            (df_engineered.loc[mask, friends_col] + epsilon)
        )
        logger.info(f"  ✓ follower_following_ratio (using {followers_col} / {friends_col})")
        logger.info(f"    Non-null values: {mask.sum()}/{len(df_engineered)}")
        
        # follower_following_diff: Absolute difference feature
        df_engineered['follower_following_diff'] = np.nan
        df_engineered.loc[mask, 'follower_following_diff'] = (
            df_engineered.loc[mask, followers_col] - 
            df_engineered.loc[mask, friends_col]
        )
        logger.info(f"  ✓ follower_following_diff")
    else:
        logger.warning("  ⚠ Could not find followers_count or friends_count columns")
        df_engineered['follower_following_ratio'] = np.nan
        df_engineered['follower_following_diff'] = np.nan
    
    # ========== 2. Behavioral Rates Features ==========
    # 行为速率特征 / Caractéristiques de taux comportemental / Behavioral Rates Features
    logger.info("\n[2] Creating Behavioral Rates Features...")
    
    # 2.1 account_age_days
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
        logger.info(f"  ✓ account_age_days (using {created_col})")
        logger.info(f"    Non-null values: {account_ages.notna().sum()}/{len(df_engineered)}")
    else:
        df_engineered['account_age_days'] = np.nan
        logger.warning("  ⚠ Could not find user.created_at column")
    
    # 2.2 tweets_per_day: Eliminates temporal collinearity
    # Formula: statuses_count / (account_age_days + epsilon)
    user_statuses_cols = [col for col in df.columns 
                         if 'user.statuses_count' in col.lower() 
                         and 'quoted_status' not in col.lower()]
    if not user_statuses_cols:
        user_statuses_cols = [col for col in df.columns if 'statuses_count' in col.lower()]
    
    if user_statuses_cols and 'account_age_days' in df_engineered.columns:
        statuses_col = user_statuses_cols[0]
        mask = df_engineered['account_age_days'].notna() & df_engineered[statuses_col].notna()
        df_engineered['tweets_per_day'] = np.nan
        df_engineered.loc[mask, 'tweets_per_day'] = (
            df_engineered.loc[mask, statuses_col] / 
            (df_engineered.loc[mask, 'account_age_days'] + epsilon)
        )
        logger.info(f"  ✓ tweets_per_day")
    
    # 2.3 listed_per_day
    user_listed_cols = [col for col in df.columns 
                       if 'user.listed_count' in col.lower() 
                       and 'quoted_status' not in col.lower()]
    if not user_listed_cols:
        user_listed_cols = [col for col in df.columns if 'listed_count' in col.lower()]
    
    if user_listed_cols and 'account_age_days' in df_engineered.columns:
        listed_col = user_listed_cols[0]
        mask = df_engineered['account_age_days'].notna() & df_engineered[listed_col].notna()
        df_engineered['listed_per_day'] = np.nan
        df_engineered.loc[mask, 'listed_per_day'] = (
            df_engineered.loc[mask, listed_col] / 
            (df_engineered.loc[mask, 'account_age_days'] + epsilon)
        )
        logger.info(f"  ✓ listed_per_day")
    
    # ========== 3. Tweet-level Interaction Features ==========
    logger.info("\n[3] Creating Tweet-level Interaction Features...")
    
    # 3.1 is_reply: Key Observer signal (核心Observer信号)
    # Observer users engage in reciprocal conversations (互惠对话)
    if 'in_reply_to_user_id' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id'].notna().astype(int)
        logger.info(f"  ✓ is_reply (key Observer predictor)")
    elif 'in_reply_to_user_id_str' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id_str'].notna().astype(int)
        logger.info(f"  ✓ is_reply (using in_reply_to_user_id_str)")
    else:
        df_engineered['is_reply'] = 0
        logger.warning("  ⚠ Could not find in_reply_to_user_id column")
    
    # 3.2 Entities counts (mentions, hashtags, URLs)
    # These are metadata counts, NOT text content (符合竞赛要求)
    
    # mentions_count
    if 'entities.user_mentions' in df_engineered.columns:
        df_engineered['mentions_count'] = df_engineered['entities.user_mentions'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        logger.info(f"  ✓ mentions_count")
    else:
        df_engineered['mentions_count'] = 0
    
    # hashtags_count
    if 'entities.hashtags' in df_engineered.columns:
        df_engineered['hashtags_count'] = df_engineered['entities.hashtags'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        logger.info(f"  ✓ hashtags_count")
    else:
        df_engineered['hashtags_count'] = 0
    
    # urls_count
    if 'entities.urls' in df_engineered.columns:
        df_engineered['urls_count'] = df_engineered['entities.urls'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        logger.info(f"  ✓ urls_count")
    else:
        df_engineered['urls_count'] = 0
    
    # ========== 4. Categorical Encoding: Source Field ==========
    # 分类特征编码 / Encodage de caractéristiques catégorielles / Categorical Encoding
    logger.info("\n[4] Encoding Categorical Features (Source)...")
    
    # Use OrdinalEncoder (NOT One-Hot Encoding) for tree models
    # Reason: Tree models handle ordinal encoding more efficiently than sparse one-hot
    # 原因：树模型处理序数编码比稀疏one-hot更高效
    
    if 'source' in df_engineered.columns:
        source_encoder = transformers.get('source_encoder')
        
        # Prepare data: fill NaN with 'unknown'
        source_data = df_engineered['source'].fillna('unknown').values.reshape(-1, 1)
        
        if is_train:
            # Training set: fit and transform
            if source_encoder is None:
                source_encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                source_encoder.fit(source_data)
                transformers['source_encoder'] = source_encoder
                logger.info(f"  ✓ Fitted source_encoder on {len(source_data)} samples")
            
            source_encoded = source_encoder.transform(source_data)
            df_engineered['source_encoded'] = source_encoded.flatten()
            logger.info(f"  ✓ source_encoded (unique values: {df_engineered['source'].nunique()})")
        else:
            # Test set: only transform (use training encoder)
            if source_encoder is None:
                logger.warning("  ⚠ No source_encoder provided for test set, creating new one (not recommended)")
                source_encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                source_encoder.fit(source_data)
                transformers['source_encoder'] = source_encoder
            
            source_encoded = source_encoder.transform(source_data)
            df_engineered['source_encoded'] = source_encoded.flatten()
            logger.info(f"  ✓ source_encoded (transformed using training encoder)")
    else:
        df_engineered['source_encoded'] = np.nan
        logger.warning("  ⚠ Could not find 'source' column")
    
    # ========== 5. Select Only Numeric Features ==========
    # Keep only numeric columns (XGBoost requires numeric input)
    # Remove text content columns and string columns (dates, etc.)
    logger.info("\n[5] Filtering to Numeric Features Only...")
    
    # Remove text content columns explicitly
    text_cols = ['text', 'extended_tweet.full_text', 
                 'quoted_status.text', 'quoted_status.extended_tweet.full_text']
    
    for col in text_cols:
        if col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[col])
            logger.debug(f"  Removed text column: {col}")
    
    # Identify non-numeric columns (object, string types)
    non_numeric_cols = []
    for col in df_engineered.columns:
        dtype = df_engineered[col].dtype
        if dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            # Entities columns are already processed into counts, so drop the original object columns
            # Also drop any other object columns (dates, strings, etc.)
            non_numeric_cols.append(col)
    
    # Remove non-numeric columns (entities are already processed into counts)
    if non_numeric_cols:
        logger.info(f"  Removing {len(non_numeric_cols)} non-numeric columns:")
        for col in non_numeric_cols[:10]:  # Show first 10
            logger.debug(f"    - {col}")
        if len(non_numeric_cols) > 10:
            logger.debug(f"    ... and {len(non_numeric_cols) - 10} more")
        df_engineered = df_engineered.drop(columns=non_numeric_cols)
    
    # Ensure all remaining columns are numeric
    # Convert any remaining object columns to numeric (if possible) or drop them
    remaining_object_cols = df_engineered.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_object_cols:
        # Try to convert to numeric
        try:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
            logger.debug(f"  Converted {col} to numeric")
        except:
            # If conversion fails, drop the column
            df_engineered = df_engineered.drop(columns=[col])
            logger.debug(f"  Dropped non-convertible column: {col}")
    
    # Final check: ensure all columns are numeric
    final_non_numeric = df_engineered.select_dtypes(exclude=[np.number]).columns.tolist()
    if final_non_numeric:
        logger.warning(f"  ⚠ Warning: {len(final_non_numeric)} non-numeric columns remain, dropping them")
        df_engineered = df_engineered.drop(columns=final_non_numeric)
    
    logger.info(f"  ✓ Final numeric columns: {len(df_engineered.columns)}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Feature Engineering Complete!")
    logger.info(f"  Original columns: {len(df.columns)}")
    logger.info(f"  Final numeric columns: {len(df_engineered.columns)}")
    logger.info(f"  Engineered features added: follower_following_ratio, account_age_days, tweets_per_day, etc.")
    logger.info(f"{'='*60}")
    
    return df_engineered, transformers


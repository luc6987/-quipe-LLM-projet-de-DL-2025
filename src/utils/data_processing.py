"""
Data Processing and Feature Engineering Utilities
数据处理与特征工程工具模块

This module consolidates data loading and feature engineering logic used across different models.
It implements the technical blueprint for metadata feature extraction and ensures consistent
processing between training and test sets.
"""

import html
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.preprocessing import OrdinalEncoder
import warnings

# Import unified path configuration
from src.utils.paths import get_data_cache_dir

# Configure logger
logger = logging.getLogger(__name__)

HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]{1,30}")
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
MULTISPACE_PATTERN = re.compile(r"\s+")
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U0001F900-\U0001F9FF"
    u"\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
DEFAULT_TEXT_COLUMNS = [
    'text',
    'extended_tweet.full_text',
    'full_text',
    'quoted_status.text',
    'user.description',
]
CLEAN_CACHE_SUFFIX = "_cleaned.pkl"


def _build_cache_path(file_path: str, cache_dir: Optional[Union[str, Path]]) -> Path:
    """
    Build cache path for cleaned dataframes.
    Uses unified data cache directory if cache_dir is not specified.
    """
    if cache_dir:
        base_dir = Path(cache_dir)
    else:
        # Use unified data cache directory
        base_dir = get_data_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    return base_dir / f"{stem}{CLEAN_CACHE_SUFFIX}"


def normalize_tweet_text(text: Union[str, float, int]) -> str:
    """
    Apply BERTweet-aligned normalization to a single tweet string.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\u200d', ' ')
    text = URL_PATTERN.sub(" HTTPURL ", text)
    text = HANDLE_PATTERN.sub(" @USER ", text)
    text = EMOJI_PATTERN.sub(" <EMOJI> ", text)
    text = text.replace("&amp;", "&")
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize target text columns in the provided DataFrame.
    """
    if columns is None:
        columns = DEFAULT_TEXT_COLUMNS
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return df
    for col in columns:
        df[col] = df[col].apply(normalize_tweet_text)
    return df


def load_data(
    file_path: str,
    clean_text: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    text_columns: Optional[List[str]] = None,
    force_recache: bool = False
) -> pd.DataFrame:
    """
    Load data from a JSONL file and normalize it into a DataFrame.
    Uses row-by-row processing to handle potential memory issues with large files.
    
    Args:
        file_path: Path to the JSONL file.
        clean_text: Whether to normalize tweet text columns.
        cache_dir: Optional directory for caching cleaned dataframes.
        text_columns: Custom list of columns to normalize; defaults to common tweet text fields.
        force_recache: When True, rebuilds the cache regardless of existing files.
        
    Returns:
        pd.DataFrame: The loaded and normalized data.
    """
    cache_path = None
    if clean_text:
        cache_path = _build_cache_path(file_path, cache_dir)
        if cache_path.exists() and not force_recache:
            logger.info(f"Loading cached cleaned dataframe from {cache_path} ...")
            return pd.read_pickle(cache_path)
    
    logger.info(f"Loading data from {file_path}...")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  Processed {i} lines...")
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"  Loaded {len(records)} records, normalizing...")
    df = json_normalize(records)
    
    if clean_text:
        logger.info("Applying tweet text normalization...")
        df = normalize_text_columns(df, columns=text_columns)
        if cache_path:
            df.to_pickle(cache_path)
            logger.info(f"Cached cleaned dataframe to {cache_path}")
    
    logger.info(f"  Data shape: {df.shape}")
    return df

def calculate_account_age(created_at_str: str) -> Optional[int]:
    """
    Calculate account age in days from Twitter created_at string.
    
    Args:
        created_at_str: Twitter created_at format string (e.g., "Wed Mar 17 12:59:53 +0000 2021")
        
    Returns:
        int: Account age in days, or None if parsing fails.
    """
    if pd.isna(created_at_str) or created_at_str is None:
        return None
    
    try:
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        now = datetime.now(created_at.tzinfo)
        age_days = (now - created_at).days
        return max(0, age_days)
    except (ValueError, TypeError):
        return None

def engineer_features(df: pd.DataFrame, is_train: bool = True, 
                     transformers: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Engineer metadata features according to the technical blueprint.
    
    Features created:
    1. Network Asymmetry: follower_following_ratio, follower_following_diff
    2. Behavioral Rates: account_age_days, tweets_per_day, listed_per_day
    3. Tweet-level Features: is_reply, mentions_count, hashtags_count, urls_count, tweet_engagement_rate
    4. Categorical Encoding: OrdinalEncoder for 'source' and other categorical columns
    
    Args:
        df: Input DataFrame.
        is_train: Whether this is training data (affects transformer fitting).
        transformers: Dictionary of transformers from training set (required for test set).
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Processed DataFrame and transformers dictionary.
    """
    logger.info("Starting Feature Engineering...")
    
    # Work on a copy to avoid side effects
    df_engineered = df.copy()
    
    if transformers is None:
        transformers = {}
    
    epsilon = 1e-6
    
    # --- 0. Pre-cleaning: Drop high-leakage / non-feature columns ---
    # Keep user.id_str/id for identification if needed, but drop strictly non-predictive or text columns
    # Note: We keep numerical counts and some IDs momentarily for feature creation
    
    # --- 1. Network Asymmetry Features ---
    logger.info("[1] Creating Network Asymmetry Features")
    
    # Find columns
    user_followers_cols = [c for c in df_engineered.columns if 'followers_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not user_followers_cols: user_followers_cols = [c for c in df_engineered.columns if 'followers_count' in c.lower()]
    
    user_friends_cols = [c for c in df_engineered.columns if 'friends_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not user_friends_cols: user_friends_cols = [c for c in df_engineered.columns if 'friends_count' in c.lower()]
    
    if user_followers_cols and user_friends_cols:
        followers_col = user_followers_cols[0]
        friends_col = user_friends_cols[0]
        
        mask = df_engineered[followers_col].notna() & df_engineered[friends_col].notna()
        
        # Ratio
        df_engineered['follower_following_ratio'] = np.nan
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            df_engineered.loc[mask, followers_col] / 
            (df_engineered.loc[mask, friends_col] + epsilon)
        )
        
        # Difference
        df_engineered['follower_following_diff'] = np.nan
        df_engineered.loc[mask, 'follower_following_diff'] = (
            df_engineered.loc[mask, followers_col] - 
            df_engineered.loc[mask, friends_col]
        )
    
    # --- 2. Behavioral Rates Features ---
    logger.info("[2] Creating Behavioral Rates Features")
    
    # Account Age
    user_created_cols = [c for c in df_engineered.columns if 'user.created_at' in c.lower() and 'quoted_status' not in c.lower()]
    if not user_created_cols: user_created_cols = [c for c in df_engineered.columns if 'created_at' in c.lower() and 'user' in c.lower()]
    
    if user_created_cols:
        created_col = user_created_cols[0]
        df_engineered['account_age_days'] = df_engineered[created_col].apply(calculate_account_age)
    else:
        df_engineered['account_age_days'] = np.nan
        
    # Tweets per day
    user_statuses_cols = [c for c in df_engineered.columns if 'user.statuses_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not user_statuses_cols: user_statuses_cols = [c for c in df_engineered.columns if 'statuses_count' in c.lower()]
    
    if user_statuses_cols and 'account_age_days' in df_engineered.columns:
        statuses_col = user_statuses_cols[0]
        mask = df_engineered['account_age_days'].notna() & df_engineered[statuses_col].notna()
        
        df_engineered['tweets_per_day'] = np.nan
        df_engineered.loc[mask, 'tweets_per_day'] = (
            df_engineered.loc[mask, statuses_col] / 
            (df_engineered.loc[mask, 'account_age_days'] + epsilon)
        )

    # Listed per day
    user_listed_cols = [c for c in df_engineered.columns if 'user.listed_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not user_listed_cols: user_listed_cols = [c for c in df_engineered.columns if 'listed_count' in c.lower()]
    
    if user_listed_cols and 'account_age_days' in df_engineered.columns:
        listed_col = user_listed_cols[0]
        mask = df_engineered['account_age_days'].notna() & df_engineered[listed_col].notna()
        
        df_engineered['listed_per_day'] = np.nan
        df_engineered.loc[mask, 'listed_per_day'] = (
            df_engineered.loc[mask, listed_col] / 
            (df_engineered.loc[mask, 'account_age_days'] + epsilon)
        )

    # --- 2.x Tweet Engagement Rate ---
    logger.info("[2.x] Creating Tweet Engagement Rate")
    retweet_cols = [c for c in df_engineered.columns if 'retweet_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not retweet_cols: retweet_cols = [c for c in df_engineered.columns if 'retweet_count' in c.lower()]
    
    favorite_cols = [c for c in df_engineered.columns if 'favorite_count' in c.lower() and 'quoted_status' not in c.lower()]
    if not favorite_cols: favorite_cols = [c for c in df_engineered.columns if 'favorite_count' in c.lower()]
    
    if retweet_cols and favorite_cols and user_followers_cols:
        retweet_col = retweet_cols[0]
        favorite_col = favorite_cols[0]
        followers_col = user_followers_cols[0] # reusing from above
        
        mask = (df_engineered[retweet_col].notna() & 
                df_engineered[favorite_col].notna() & 
                df_engineered[followers_col].notna())
                
        df_engineered['tweet_engagement_rate'] = np.nan
        df_engineered.loc[mask, 'tweet_engagement_rate'] = (
            (df_engineered.loc[mask, retweet_col] + df_engineered.loc[mask, favorite_col]) /
            (df_engineered.loc[mask, followers_col] + epsilon)
        )

    # --- 3. Tweet-level Interaction Features ---
    logger.info("[3] Creating Tweet-level Interaction Features")
    
    # is_reply
    if 'in_reply_to_user_id' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id'].notna().astype(int)
    elif 'in_reply_to_user_id_str' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_user_id_str'].notna().astype(int)
    elif 'in_reply_to_screen_name' in df_engineered.columns:
        df_engineered['is_reply'] = df_engineered['in_reply_to_screen_name'].notna().astype(int)
    else:
        df_engineered['is_reply'] = 0
        
    # Entity counts
    for entity, col_name in [('user_mentions', 'mentions_count'), ('hashtags', 'hashtags_count'), ('urls', 'urls_count')]:
        full_col = f'entities.{entity}'
        if full_col in df_engineered.columns:
            df_engineered[col_name] = df_engineered[full_col].apply(
                lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
            )
        else:
            df_engineered[col_name] = 0

    # --- 4. Categorical Encoding (Source) ---
    logger.info("[4] Encoding Source")
    if 'source' in df_engineered.columns:
        source_data = df_engineered['source'].fillna('unknown').values.reshape(-1, 1)
        
        if is_train:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(source_data)
            transformers['source_encoder'] = encoder
        else:
            if 'source_encoder' not in transformers:
                # Fallback if missing in test
                transformers['source_encoder'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                transformers['source_encoder'].fit(source_data)
            encoder = transformers['source_encoder']
            
        df_engineered['source_encoded'] = encoder.transform(source_data).flatten()
        df_engineered.drop(columns=['source'], inplace=True)

    # --- 5. Cleanup: Drop High Cardinality / Text / Leakage columns ---
    logger.info("[5] Dropping unused columns")
    
    # Define drop patterns or list
    drop_cols = [
        'text', 'extended_tweet.full_text', 'quoted_status.text',
        'user.description', 'user.name', 'user.screen_name', 'user.location', 'user.url',
        'place.full_name', 'place.name', 'place.url', 'place.id',
        'created_at', 'quoted_status.created_at',
        'in_reply_to_user_id', 'in_reply_to_status_id', 'in_reply_to_screen_name',
        'in_reply_to_user_id_str', 'in_reply_to_status_id_str',
        'quoted_status_id', 'quoted_status_id_str',
        'entities.user_mentions', 'entities.hashtags', 'entities.urls', 'entities.media', 'entities.symbols',
        'extended_entities.media',
        'display_text_range'
    ]
    
    # Drop columns that match drop list or contain certain substrings (like 'profile_image')
    cols_to_drop = []
    for col in df_engineered.columns:
        if col in drop_cols:
            cols_to_drop.append(col)
            continue
        if 'profile_' in col and 'color' not in col: # Drop image URLs etc, keep colors if useful (optional, usually drop all)
             cols_to_drop.append(col)
        if 'url' in col and col not in ['urls_count']: # Drop other URL fields
             cols_to_drop.append(col)
        if 'id_str' in col and col != 'challenge_id': # Drop ID strings except challenge_id (needed for submission)
             cols_to_drop.append(col)
        if col.startswith('quoted_status'): # Simplify by dropping deep quoted status details not used
             cols_to_drop.append(col)

    # Also drop columns created above that we no longer need (like raw created_at)
    if user_created_cols: cols_to_drop.extend([c for c in user_created_cols if c in df_engineered.columns])

    df_engineered.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')

    # --- 6. General Categorical Encoding for remaining objects ---
    logger.info("[6] Encoding remaining categorical features")
    
    cat_cols = df_engineered.select_dtypes(include=['object', 'string']).columns.tolist()
    # exclude challenge_id and label from encoding if present
    cat_cols = [c for c in cat_cols if c not in ['challenge_id', 'label']]
    
    if cat_cols:
        cat_data = df_engineered[cat_cols].fillna('unknown').astype(str).values
        
        if is_train:
            cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            cat_encoder.fit(cat_data)
            transformers['cat_encoder'] = cat_encoder
            transformers['cat_cols'] = cat_cols
        else:
            # Align columns with training
            train_cat_cols = transformers.get('cat_cols', [])
            
            # Add missing cols
            for c in train_cat_cols:
                if c not in df_engineered.columns:
                    df_engineered[c] = 'unknown'
            
            # Drop extra cols
            extra = [c for c in cat_cols if c not in train_cat_cols]
            if extra:
                df_engineered.drop(columns=extra, inplace=True)
                
            # Re-order
            cat_cols = train_cat_cols
            cat_data = df_engineered[cat_cols].fillna('unknown').astype(str).values
            
            if 'cat_encoder' not in transformers:
                 # Should not happen if pipeline is correct
                 logger.warning("Categorical encoder missing for test set!")
                 transformers['cat_encoder'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                 transformers['cat_encoder'].fit(cat_data)
            
            cat_encoder = transformers['cat_encoder']

        encoded = cat_encoder.transform(cat_data)
        for i, col in enumerate(cat_cols):
            df_engineered[col] = encoded[:, i]

    # Final cleanup of non-numeric (except challenge_id)
    final_drop = []
    for col in df_engineered.columns:
        if col != 'challenge_id' and not pd.api.types.is_numeric_dtype(df_engineered[col]):
             final_drop.append(col)
    
    if final_drop:
        df_engineered.drop(columns=final_drop, inplace=True)

    logger.info(f"Feature Engineering Complete. Final shape: {df_engineered.shape}")
    return df_engineered, transformers

def prepare_xy(df: pd.DataFrame, target_col: str = 'label', id_col: str = 'challenge_id') -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """
    Split DataFrame into X (features), y (target), and IDs.
    """
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()
        
    if id_col in X.columns:
        ids = X[id_col]
        X = X.drop(columns=[id_col])
    else:
        ids = None
        
    return X, y, ids


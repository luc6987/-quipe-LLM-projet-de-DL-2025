"""
Feature Engineering Module
特征工程模块 / Module d'ingénierie des caractéristiques / Feature Engineering Module / Feature-Engineering-Modul

Implements feature engineering based on optimized feature processing plan.
基于优化后的特征处理计划实现特征工程。

Key Features:
1. Network Asymmetry (网络不对称性 / Asymétrie du réseau / Network Asymmetry / Netzwerk-Asymmetrie)
2. Behavioral Rates (行为速率 / Taux comportemental / Behavioral Rates / Verhaltensraten)
3. Tweet-level Interaction Features
4. Feature processing based on feature_processing_plan.json
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
import json
import os
import logging
import re
from html import unescape

# Import text embedding module
try:
    from .text_embedding import process_text_embeddings
    TEXT_EMBEDDING_AVAILABLE = True
except ImportError:
    try:
        from text_embedding import process_text_embeddings
        TEXT_EMBEDDING_AVAILABLE = True
    except ImportError:
        TEXT_EMBEDDING_AVAILABLE = False
        logger.warning("text_embedding module not available. Text embedding will be skipped.")

logger = logging.getLogger(__name__)


def load_feature_plan(plan_path: Optional[str] = None) -> Dict:
    """
    Load feature processing plan from JSON file.
    从JSON文件加载特征处理计划。
    
    Args:
        plan_path: Path to feature_processing_plan.json. If None, uses default path.
        
    Returns:
        Dictionary mapping feature names to processing instructions.
    """
    if plan_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plan_path = os.path.join(current_dir, '..', 'feature_processing_plan.json')
    
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"Feature processing plan not found: {plan_path}")
    
    with open(plan_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)
    
    logger.info(f"Loaded feature processing plan: {len(plan)} features")
    return plan


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


def extract_date_components(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Extract date components from timestamp column.
    从时间戳列提取日期组件。
    
    Args:
        df: DataFrame
        col: Column name containing timestamp
        
    Returns:
        DataFrame with new date component columns
    """
    if col not in df.columns:
        return df
    
    new_cols = {}
    
    def parse_timestamp(val):
        if pd.isna(val) or val is None:
            return None, None, None, None, None, None
        
        try:
            # Try Twitter format first
            dt = datetime.strptime(str(val), "%a %b %d %H:%M:%S %z %Y")
        except (ValueError, TypeError):
            try:
                # Try ISO format
                dt = pd.to_datetime(val)
            except (ValueError, TypeError):
                return None, None, None, None, None, None
        
        return dt.year, dt.month, dt.day, dt.hour, dt.weekday(), dt.isoweekday()
    
    results = df[col].apply(parse_timestamp)
    new_cols[f'{col}_year'] = [r[0] for r in results]
    new_cols[f'{col}_month'] = [r[1] for r in results]
    new_cols[f'{col}_day'] = [r[2] for r in results]
    new_cols[f'{col}_hour'] = [r[3] for r in results]
    new_cols[f'{col}_weekday'] = [r[4] for r in results]
    new_cols[f'{col}_isoweekday'] = [r[5] for r in results]
    
    for new_col, values in new_cols.items():
        df[new_col] = values
    
    return df


def frequency_encoding(df: pd.DataFrame, col: str, is_train: bool, 
                      transformers: Dict, feature_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply frequency encoding to a categorical column.
    对类别列应用频率编码。
    
    Args:
        df: DataFrame
        col: Column name to encode
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    transformer_key = f'freq_enc_{feature_name}'
    
    # Convert all values to string to handle mixed types, lists, arrays, and ensure hashability
    # This is safer than checking types individually
    def safe_to_string(x):
        """Safely convert any value to string, handling NaN, lists, arrays, etc."""
        try:
            # Handle NaN first
            if pd.isna(x):
                return 'nan'
            # Handle numpy arrays (including empty ones)
            if isinstance(x, np.ndarray):
                return str(x.tolist()) if x.size > 0 else '[]'
            # Handle lists (including empty ones)
            if isinstance(x, list):
                return str(x) if len(x) > 0 else '[]'
            # For other types, convert to string
            return str(x)
        except Exception:
            # Ultimate fallback
            return str(x) if x is not None else 'nan'
    
    if is_train:
        # Convert all values to string representation for frequency encoding
        col_str = df[col].apply(safe_to_string)
        
        # Calculate frequency mapping
        value_counts = col_str.value_counts()
        freq_map = value_counts.to_dict()
        transformers[transformer_key] = freq_map
        
        # Apply frequency encoding
        df[col] = col_str.map(freq_map).fillna(0)
        logger.debug(f"  Frequency encoded {col}: {len(freq_map)} unique values")
    else:
        # Use training frequency map
        freq_map = transformers.get(transformer_key)
        if freq_map is None:
            logger.warning(f"  No frequency map for {col}, using 0 for all values")
            df[col] = 0
        else:
            # Convert to string for mapping (using same safe conversion)
            col_str = df[col].apply(safe_to_string)
            df[col] = col_str.map(freq_map).fillna(0)
    
    return df, transformers


def ordinal_encoding(df: pd.DataFrame, col: str, is_train: bool,
                     transformers: Dict, feature_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply ordinal encoding to a categorical column.
    对类别列应用序数编码。
    
    Args:
        df: DataFrame
        col: Column name to encode
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    transformer_key = f'ord_enc_{feature_name}'
    
    # Prepare data: fill NaN with 'unknown'
    col_data = df[col].fillna('unknown').astype(str).values.reshape(-1, 1)
    
    if is_train:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        encoder.fit(col_data)
        transformers[transformer_key] = encoder
        encoded = encoder.transform(col_data)
        df[col] = encoded.flatten()
    else:
        encoder = transformers.get(transformer_key)
        if encoder is None:
            logger.warning(f"  No encoder for {col}, using -1 for all values")
            df[col] = -1
        else:
            encoded = encoder.transform(col_data)
            df[col] = encoded.flatten()
    
    return df, transformers


def bool_to_int(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert boolean column to integer (0/1).
    将布尔列转换为整数（0/1）。
    
    Args:
        df: DataFrame
        col: Column name to convert
        
    Returns:
        DataFrame with converted column
    """
    if col not in df.columns:
        return df
    
    # Handle NaN values and convert to boolean first, then to int
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        # String/object type: convert True/False strings to boolean
        # Handle string representations of booleans
        def str_to_bool(val):
            if pd.isna(val) or val is None:
                return False
            if isinstance(val, str):
                val_lower = val.lower().strip()
                if val_lower in ('true', '1', 'yes', 't'):
                    return True
                elif val_lower in ('false', '0', 'no', 'f', ''):
                    return False
            return bool(val)
        
        df[col] = df[col].apply(str_to_bool)
    else:
        # Numeric or boolean type: fill NaN with False
        df[col] = df[col].fillna(False)
        # Ensure it's boolean type
        if not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(bool)
    
    # Convert boolean to int (0 or 1)
    df[col] = df[col].astype(int)
    return df


def impute_median(df: pd.DataFrame, col: str, is_train: bool,
                  transformers: Dict, feature_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values with median.
    用中位数填充缺失值。
    
    Args:
        df: DataFrame
        col: Column name to impute
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    transformer_key = f'median_{feature_name}'
    
    if is_train:
        median_val = df[col].median()
        transformers[transformer_key] = median_val
        df[col] = df[col].fillna(median_val)
    else:
        median_val = transformers.get(transformer_key)
        if median_val is None:
            logger.warning(f"  No median value for {col}, using 0")
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(median_val)
    
    return df, transformers


def extract_binary_flag(df: pd.DataFrame, col: str, new_col_name: Optional[str] = None) -> pd.DataFrame:
    """
    Extract binary flag from a column (NaN -> 0, Value -> 1).
    Core signal: NaN means broadcast, Value means reply.
    从列中提取二进制标志（NaN -> 0, 有值 -> 1）。
    核心信号：NaN表示广播，有值表示回复。
    
    Args:
        df: DataFrame
        col: Column name to extract flag from
        new_col_name: Name for the new binary column. If None, uses f'has_{col}'
        
    Returns:
        DataFrame with new binary column
    """
    if col not in df.columns:
        return df
    
    if new_col_name is None:
        new_col_name = f'has_{col}'
    
    # NaN -> 0 (broadcast), Value -> 1 (reply)
    df[new_col_name] = df[col].notna().astype(int)
    return df


def clean_and_categorize_source(df: pd.DataFrame, col: str, is_train: bool,
                                transformers: Dict, feature_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean HTML tags from source and categorize as Mobile/Web/Pro Tools/Other.
    清洗source字段的HTML标签，并分类为Mobile/Web/Pro Tools/Other。
    
    Categories:
    - Mobile: Twitter for iPhone/Android
    - Professional Tools: TweetDeck, Buffer, Hootsuite
    - Web: Twitter Web Client
    - Other: Everything else
    
    Args:
        df: DataFrame
        col: Column name containing source
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    transformer_key = f'source_cat_{feature_name}'
    new_col = f'{col}_category'
    
    def clean_source(x):
        """Clean HTML and categorize source."""
        if pd.isna(x) or x is None:
            return 'unknown'
        
        # Clean HTML tags and unescape HTML entities
        x_str = str(x)
        # Remove HTML tags
        x_str = re.sub(r'<[^>]+>', '', x_str)
        # Unescape HTML entities
        x_str = unescape(x_str)
        x_lower = x_str.lower()
        
        # Categorize
        if 'iphone' in x_lower or 'android' in x_lower:
            return 'mobile'
        elif 'web' in x_lower or 'twitter.com' in x_lower:
            return 'web'
        elif any(tool in x_lower for tool in ['tweetdeck', 'buffer', 'hootsuite', 'sprout social', 'socialflow']):
            return 'professional_tool'
        else:
            return 'other'
    
    # Apply cleaning and categorization
    df[new_col] = df[col].apply(clean_source)
    
    # Ordinal encode the category
    if is_train:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        col_data = df[new_col].fillna('unknown').astype(str).values.reshape(-1, 1)
        encoder.fit(col_data)
        transformers[transformer_key] = encoder
        encoded = encoder.transform(col_data)
        df[new_col] = encoded.flatten()
    else:
        encoder = transformers.get(transformer_key)
        if encoder is None:
            logger.warning(f"  No encoder for {new_col}, using -1 for all values")
            df[new_col] = -1
        else:
            col_data = df[new_col].fillna('unknown').astype(str).values.reshape(-1, 1)
            encoded = encoder.transform(col_data)
            df[new_col] = encoded.flatten()
    
    # Drop original source column
    if col in df.columns:
        df = df.drop(columns=[col])
    
    return df, transformers


def extract_text_metadata(df: pd.DataFrame, col: str, keep_original: bool = False) -> pd.DataFrame:
    """
    Extract metadata from text column: length, mention count, hashtag count, URL presence.
    从文本列提取元数据：长度、提及数量、标签数量、URL存在性。
    
    Args:
        df: DataFrame
        col: Column name containing text
        keep_original: If True, keep original text column (for embedding extraction later)
        
    Returns:
        DataFrame with new metadata columns
    """
    if col not in df.columns:
        return df
    
    # Text length
    df[f'{col}_len'] = df[col].astype(str).apply(len)
    
    # Mention count (@)
    df[f'{col}_mention_count'] = df[col].astype(str).apply(lambda x: x.count('@'))
    
    # Hashtag count (#)
    df[f'{col}_hashtag_count'] = df[col].astype(str).apply(lambda x: x.count('#'))
    
    # URL presence
    df[f'{col}_has_url'] = df[col].astype(str).str.contains('http', na=False).astype(int)
    
    # Drop original text column only if not keeping for embedding
    if not keep_original and col in df.columns:
        df = df.drop(columns=[col])
    
    return df


def extract_description_metadata(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Extract metadata from user.description: length, email presence, official words.
    从user.description提取元数据：长度、email存在性、官方词汇。
    
    Args:
        df: DataFrame
        col: Column name containing description
        
    Returns:
        DataFrame with new metadata columns
    """
    if col not in df.columns:
        return df
    
    # Description length
    df['desc_len'] = df[col].astype(str).apply(len)
    
    # Email presence (business contact flag -> influencer indicator)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    df['has_email'] = df[col].astype(str).str.contains(email_pattern, na=False, regex=True).astype(int)
    
    # Official words (Official, Page, Author, etc.)
    official_words = ['official', 'page', 'author', 'verified', 'account', 'team', 'company', 'brand']
    pattern = '|'.join([f'\\b{word}\\b' for word in official_words])
    df['has_official_words'] = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True).astype(int)
    
    # Drop original description column
    if col in df.columns:
        df = df.drop(columns=[col])
    
    return df


def extract_location_metadata(df: pd.DataFrame, col: str, is_train: bool,
                             transformers: Dict, feature_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract metadata from user.location: has_location flag, country/city extraction.
    从user.location提取元数据：has_location标志、国家/城市提取。
    
    Args:
        df: DataFrame
        col: Column name containing location
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    # Has location flag
    df['has_location'] = df[col].notna().astype(int)
    
    # Extract country/city (simplified: extract first meaningful word/phrase)
    # For now, we'll use frequency encoding of the location string itself
    # In production, you might want to use a geocoding API
    transformer_key = f'location_freq_{feature_name}'
    
    if is_train:
        # Frequency encoding of location strings
        location_counts = df[col].value_counts()
        freq_map = location_counts.to_dict()
        transformers[transformer_key] = freq_map
        df['location_freq'] = df[col].map(freq_map).fillna(0)
    else:
        freq_map = transformers.get(transformer_key)
        if freq_map is None:
            logger.warning(f"  No frequency map for {col}, using 0 for all values")
            df['location_freq'] = 0
        else:
            df['location_freq'] = df[col].map(freq_map).fillna(0)
    
    # Drop original location column
    if col in df.columns:
        df = df.drop(columns=[col])
    
    return df, transformers


def target_encoding_cv(df: pd.DataFrame, col: str, y: pd.Series, n_splits: int = 5,
                       transformers: Dict = None, feature_name: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply target encoding with cross-validation to avoid overfitting.
    使用交叉验证进行目标编码以避免过拟合。
    
    Args:
        df: DataFrame
        col: Column name to encode
        y: Target variable (labels)
        n_splits: Number of folds for cross-validation
        transformers: Dictionary of transformers
        feature_name: Original feature name (for transformer key)
        
    Returns:
        Tuple of (DataFrame, updated transformers dict)
    """
    if col not in df.columns:
        return df, transformers
    
    if feature_name is None:
        feature_name = col
    
    transformer_key = f'target_enc_{feature_name}'
    new_col = f'{col}_target_enc'
    
    # Initialize the encoded column
    df[new_col] = 0.0
    
    # Calculate global mean for smoothing
    global_mean = y.mean()
    
    # Create a temporary DataFrame with the column and target for easier groupby
    temp_df = pd.DataFrame({col: df[col], 'target': y.values})
    
    # Cross-validation approach
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        # Calculate target mean for each category in training fold
        train_temp = temp_df.iloc[train_idx]
        
        # Group by category and calculate mean
        category_means = train_temp.groupby(col)['target'].mean()
        
        # Apply to validation fold
        val_temp = temp_df.iloc[val_idx]
        df.loc[val_idx, new_col] = val_temp[col].map(category_means).fillna(global_mean)
    
    # For test set, use the full training set statistics
    if transformers is not None:
        # Calculate full training statistics
        full_category_means = temp_df.groupby(col)['target'].mean()
        transformers[transformer_key] = {
            'category_means': full_category_means.to_dict(),
            'global_mean': global_mean
        }
    
    # Drop original column
    if col in df.columns:
        df = df.drop(columns=[col])
    
    return df, transformers


def engineer_features(
    df: pd.DataFrame,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    kaggle_max: bool = False,
    plan_path: Optional[str] = None,
    y_train: Optional[pd.Series] = None,
    use_text_embedding: bool = True,
    text_embedding_n_components: int = 16,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Engineer features according to the optimized feature processing plan.
    根据优化后的特征处理计划进行特征工程。
    
    Args:
        df: Input DataFrame (flattened JSON)
        is_train: Whether this is training data (affects transformer fitting)
        transformers: Dictionary of transformers from training set (for test set)
        kaggle_max: Whether to use kaggle_max mode (currently not used in plan-based approach)
        plan_path: Path to feature_processing_plan.json
        y_train: Target variable (required for target encoding in training mode)
        use_text_embedding: Whether to extract BERTweet embeddings (default: True)
        text_embedding_n_components: Number of PCA components for text embedding (default: 16)
        
    Returns:
        Tuple of:
        - df_engineered: DataFrame with engineered features
        - transformers: Dictionary of transformers
    """
    logger.info("="*60)
    logger.info("Feature Engineering: Processing Features Based on Plan")
    logger.info("="*60)
    
    df_engineered = df.copy()
    
    # Load feature processing plan
    feature_plan = load_feature_plan(plan_path)
    
    if transformers is None:
        transformers = {}
    
    # Store original text columns for embedding extraction (before dropping)
    text_cols_for_embedding = []
    if use_text_embedding and TEXT_EMBEDDING_AVAILABLE:
        text_cols_for_embedding = [col for col in df_engineered.columns 
                                   if col in ['text', 'extended_tweet.full_text']]
    
    # Always drop challenge_id and label if present
    drop_cols = ['challenge_id', 'label']
    for col in drop_cols:
        if col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[col])
    
    # Process features according to plan
    processed_count = {'drop': 0, 'keep': 0, 'transform': 0}
    
    # Group features by action and method for batch processing
    drop_features = []
    keep_passthrough = []
    keep_bool_to_int = []
    keep_impute_median = []
    transform_frequency = []
    transform_ordinal = []
    transform_extract_date = []
    transform_extract_account_age = []
    transform_extract_binary_flag = []
    transform_clean_source = []
    transform_extract_text_metadata = []
    transform_extract_description_metadata = []
    transform_extract_location_metadata = []
    transform_target_encoding = []
    
    for feature_name, plan_info in feature_plan.items():
        action = plan_info.get('action', 'drop')
        method = plan_info.get('method', 'none')
        
        if feature_name not in df_engineered.columns:
            continue
        
        if action == 'drop':
            drop_features.append(feature_name)
        elif action == 'keep':
            if method == 'passthrough':
                keep_passthrough.append(feature_name)
            elif method == 'bool_to_int':
                keep_bool_to_int.append(feature_name)
            elif method == 'impute_median':
                keep_impute_median.append(feature_name)
        elif action == 'transform':
            if method == 'frequency_encoding':
                transform_frequency.append(feature_name)
            elif method == 'ordinal_encoding':
                transform_ordinal.append(feature_name)
            elif method == 'extract_date_components':
                transform_extract_date.append(feature_name)
            elif method == 'extract_account_age':
                transform_extract_account_age.append(feature_name)
            elif method == 'extract_binary_flag':
                transform_extract_binary_flag.append(feature_name)
            elif method == 'clean_and_categorize_source':
                transform_clean_source.append(feature_name)
            elif method == 'extract_text_metadata':
                transform_extract_text_metadata.append(feature_name)
            elif method == 'extract_description_metadata':
                transform_extract_description_metadata.append(feature_name)
            elif method == 'extract_location_metadata':
                transform_extract_location_metadata.append(feature_name)
            elif method == 'target_encoding':
                transform_target_encoding.append(feature_name)
    
    # Drop features
    if drop_features:
        logger.info(f"\n[1] Dropping {len(drop_features)} features...")
        df_engineered = df_engineered.drop(columns=drop_features, errors='ignore')
        processed_count['drop'] = len(drop_features)
    
    # Keep passthrough (numeric features, no processing needed)
    if keep_passthrough:
        logger.info(f"\n[2] Keeping {len(keep_passthrough)} numeric features (passthrough)...")
        processed_count['keep'] += len(keep_passthrough)
    
    # Convert boolean to int
    if keep_bool_to_int:
        logger.info(f"\n[3] Converting {len(keep_bool_to_int)} boolean features to int...")
        for col in keep_bool_to_int:
            df_engineered = bool_to_int(df_engineered, col)
        processed_count['keep'] += len(keep_bool_to_int)
    
    # Impute median for numeric features with missing values
    if keep_impute_median:
        logger.info(f"\n[4] Imputing median for {len(keep_impute_median)} numeric features...")
        for col in keep_impute_median:
            df_engineered, transformers = impute_median(
                df_engineered, col, is_train, transformers, col
            )
        processed_count['keep'] += len(keep_impute_median)
    
    # Frequency encoding
    if transform_frequency:
        logger.info(f"\n[5] Frequency encoding {len(transform_frequency)} categorical features...")
        for col in transform_frequency:
            df_engineered, transformers = frequency_encoding(
                df_engineered, col, is_train, transformers, col
            )
        processed_count['transform'] += len(transform_frequency)
    
    # Ordinal encoding
    if transform_ordinal:
        logger.info(f"\n[6] Ordinal encoding {len(transform_ordinal)} categorical features...")
        for col in transform_ordinal:
            df_engineered, transformers = ordinal_encoding(
                df_engineered, col, is_train, transformers, col
            )
        processed_count['transform'] += len(transform_ordinal)
    
    # Extract date components
    if transform_extract_date:
        logger.info(f"\n[7] Extracting date components from {len(transform_extract_date)} timestamp features...")
        for col in transform_extract_date:
            df_engineered = extract_date_components(df_engineered, col)
            # Drop original timestamp column after extraction
            if col in df_engineered.columns:
                df_engineered = df_engineered.drop(columns=[col])
        processed_count['transform'] += len(transform_extract_date)
    
    # Extract account age
    if transform_extract_account_age:
        logger.info(f"\n[8] Extracting account age from {len(transform_extract_account_age)} timestamp features...")
        for col in transform_extract_account_age:
            if col in df_engineered.columns:
                account_ages = df_engineered[col].apply(calculate_account_age)
                # Create new column name
                new_col = col.replace('created_at', 'account_age_days')
                df_engineered[new_col] = account_ages
                # Drop original timestamp column
                df_engineered = df_engineered.drop(columns=[col])
                logger.info(f"  ✓ Extracted {new_col} from {col}")
        processed_count['transform'] += len(transform_extract_account_age)
    
    # Extract binary flags (e.g., is_reply from in_reply_to_status_id)
    if transform_extract_binary_flag:
        logger.info(f"\n[9] Extracting binary flags from {len(transform_extract_binary_flag)} features...")
        for col in transform_extract_binary_flag:
            if col in df_engineered.columns:
                # Special handling for in_reply_to_status_id -> is_reply
                if 'in_reply_to_status_id' in col:
                    df_engineered = extract_binary_flag(df_engineered, col, 'is_reply')
                else:
                    df_engineered = extract_binary_flag(df_engineered, col)
                # Drop original column
                if col in df_engineered.columns:
                    df_engineered = df_engineered.drop(columns=[col])
                logger.info(f"  ✓ Extracted binary flag from {col}")
        processed_count['transform'] += len(transform_extract_binary_flag)
    
    # Clean and categorize source
    if transform_clean_source:
        logger.info(f"\n[10] Cleaning and categorizing {len(transform_clean_source)} source features...")
        for col in transform_clean_source:
            if col in df_engineered.columns:
                df_engineered, transformers = clean_and_categorize_source(
                    df_engineered, col, is_train, transformers, col
                )
                logger.info(f"  ✓ Cleaned and categorized {col}")
        processed_count['transform'] += len(transform_clean_source)
    
    # Extract text metadata (keep original text for embedding extraction later)
    if transform_extract_text_metadata:
        logger.info(f"\n[11] Extracting metadata from {len(transform_extract_text_metadata)} text features...")
        for col in transform_extract_text_metadata:
            if col in df_engineered.columns:
                # Keep original text column if text embedding will be used
                keep_original = use_text_embedding and TEXT_EMBEDDING_AVAILABLE
                df_engineered = extract_text_metadata(df_engineered, col, keep_original=keep_original)
                logger.info(f"  ✓ Extracted metadata from {col}" + (" (keeping original for embedding)" if keep_original else ""))
        processed_count['transform'] += len(transform_extract_text_metadata)
    
    # Extract description metadata
    if transform_extract_description_metadata:
        logger.info(f"\n[12] Extracting metadata from {len(transform_extract_description_metadata)} description features...")
        for col in transform_extract_description_metadata:
            if col in df_engineered.columns:
                df_engineered = extract_description_metadata(df_engineered, col)
                logger.info(f"  ✓ Extracted metadata from {col}")
        processed_count['transform'] += len(transform_extract_description_metadata)
    
    # Extract location metadata
    if transform_extract_location_metadata:
        logger.info(f"\n[13] Extracting metadata from {len(transform_extract_location_metadata)} location features...")
        for col in transform_extract_location_metadata:
            if col in df_engineered.columns:
                df_engineered, transformers = extract_location_metadata(
                    df_engineered, col, is_train, transformers, col
                )
                logger.info(f"  ✓ Extracted metadata from {col}")
        processed_count['transform'] += len(transform_extract_location_metadata)
    
    # Target encoding (requires y_train for training mode)
    if transform_target_encoding:
        logger.info(f"\n[14] Applying target encoding to {len(transform_target_encoding)} features...")
        if is_train:
            if y_train is None:
                logger.warning("  ⚠ Target encoding requires y_train but it's None. Skipping target encoding.")
            else:
                for col in transform_target_encoding:
                    if col in df_engineered.columns:
                        df_engineered, transformers = target_encoding_cv(
                            df_engineered, col, y_train, n_splits=5, 
                            transformers=transformers, feature_name=col
                        )
                        logger.info(f"  ✓ Target encoded {col}")
                processed_count['transform'] += len(transform_target_encoding)
        else:
            # For test set, use training statistics
            for col in transform_target_encoding:
                if col in df_engineered.columns:
                    transformer_key = f'target_enc_{col}'
                    enc_stats = transformers.get(transformer_key)
                    if enc_stats is None:
                        logger.warning(f"  ⚠ No target encoding stats for {col}, using global mean")
                        df_engineered[f'{col}_target_enc'] = 0.5  # Default to neutral
                    else:
                        category_means = enc_stats['category_means']
                        global_mean = enc_stats['global_mean']
                        df_engineered[f'{col}_target_enc'] = df_engineered[col].map(category_means).fillna(global_mean)
                    
                    # Drop original column
                    if col in df_engineered.columns:
                        df_engineered = df_engineered.drop(columns=[col])
                    logger.info(f"  ✓ Applied target encoding to {col} (test mode)")
    
    # ========== Create Engineered Features (Network Asymmetry, Behavioral Rates, etc.) ==========
    logger.info("\n[14] Creating engineered features (Network Asymmetry, Behavioral Rates, etc.)...")
    
    epsilon = 1e-6
    
    # Network Asymmetry Features
    user_followers_cols = [col for col in df_engineered.columns 
                          if 'followers_count' in col.lower() 
                          and 'quoted_status' not in col.lower()]
    user_friends_cols = [col for col in df_engineered.columns 
                        if 'friends_count' in col.lower() 
                        and 'quoted_status' not in col.lower()]
    
    if user_followers_cols and user_friends_cols:
        followers_col = user_followers_cols[0]
        friends_col = user_friends_cols[0]
        
        mask = df_engineered[followers_col].notna() & df_engineered[friends_col].notna()
        df_engineered['follower_following_ratio'] = np.nan
        df_engineered.loc[mask, 'follower_following_ratio'] = (
            df_engineered.loc[mask, followers_col] / 
            (df_engineered.loc[mask, friends_col] + epsilon)
        )
        logger.info(f"  ✓ follower_following_ratio")
        
        df_engineered['follower_following_diff'] = np.nan
        df_engineered.loc[mask, 'follower_following_diff'] = (
            df_engineered.loc[mask, followers_col] - 
            df_engineered.loc[mask, friends_col]
        )
        logger.info(f"  ✓ follower_following_diff")
    
    # Behavioral Rates Features
    account_age_cols = [col for col in df_engineered.columns if 'account_age_days' in col.lower()]
    if account_age_cols:
        account_age_col = account_age_cols[0]
        
        # tweets_per_day
        user_statuses_cols = [col for col in df_engineered.columns 
                             if 'statuses_count' in col.lower() 
                             and 'quoted_status' not in col.lower()]
        if user_statuses_cols:
            statuses_col = user_statuses_cols[0]
            mask = df_engineered[account_age_col].notna() & df_engineered[statuses_col].notna()
            df_engineered['tweets_per_day'] = np.nan
            df_engineered.loc[mask, 'tweets_per_day'] = (
                df_engineered.loc[mask, statuses_col] / 
                (df_engineered.loc[mask, account_age_col] + epsilon)
            )
            logger.info(f"  ✓ tweets_per_day")
        
        # listed_per_day
        user_listed_cols = [col for col in df_engineered.columns 
                           if 'listed_count' in col.lower() 
                           and 'quoted_status' not in col.lower()]
        if user_listed_cols:
            listed_col = user_listed_cols[0]
            mask = df_engineered[account_age_col].notna() & df_engineered[listed_col].notna()
            df_engineered['listed_per_day'] = np.nan
            df_engineered.loc[mask, 'listed_per_day'] = (
                df_engineered.loc[mask, listed_col] / 
                (df_engineered.loc[mask, account_age_col] + epsilon)
            )
            logger.info(f"  ✓ listed_per_day")
    
    # ========== Log Transform for Long-Tail Features ==========
    logger.info("\n[15] Applying log1p transformation to long-tail features...")
    
    log_transform_cols = ['retweet_count', 'favorite_count', 'user.listed_count', 'user.favourites_count']
    for col in log_transform_cols:
        # Check if column exists (might have different naming)
        matching_cols = [c for c in df_engineered.columns if col.lower() in c.lower() and 'quoted_status' not in c.lower()]
        if matching_cols:
            orig_col = matching_cols[0]
            log_col = f'log_{orig_col}'
            df_engineered[log_col] = np.log1p(df_engineered[orig_col].fillna(0))
            logger.info(f"  ✓ {log_col}")
    
    # ========== Ghost Network Reconstruction Features ==========
    logger.info("\n[16] Creating ghost network reconstruction features...")
    
    # A. Implied Virality Score: log(retweet_count + 1) * log(favorite_count + 1)
    retweet_cols = [col for col in df_engineered.columns if 'retweet_count' in col.lower() and 'quoted_status' not in col.lower()]
    favorite_cols = [col for col in df_engineered.columns if 'favorite_count' in col.lower() and 'quoted_status' not in col.lower()]
    
    if retweet_cols and favorite_cols:
        retweet_col = retweet_cols[0]
        favorite_col = favorite_cols[0]
        
        # Apply log1p transformation first
        retweet_log = np.log1p(df_engineered[retweet_col].fillna(0))
        favorite_log = np.log1p(df_engineered[favorite_col].fillna(0))
        
        df_engineered['implied_virality_score'] = retweet_log * favorite_log
        logger.info(f"  ✓ implied_virality_score")
    
    # B. Engagement Efficiency: (retweet_count + favorite_count) / (user.statuses_count + 1)
    user_statuses_cols = [col for col in df_engineered.columns 
                          if 'statuses_count' in col.lower() 
                          and 'quoted_status' not in col.lower()]
    
    if retweet_cols and favorite_cols and user_statuses_cols:
        statuses_col = user_statuses_cols[0]
        retweet_col = retweet_cols[0]
        favorite_col = favorite_cols[0]
        
        total_engagement = df_engineered[retweet_col].fillna(0) + df_engineered[favorite_col].fillna(0)
        df_engineered['ratios_engagement_efficiency'] = total_engagement / (df_engineered[statuses_col].fillna(0) + 1)
        logger.info(f"  ✓ ratios_engagement_efficiency")
    
    # C. Listed to Active Ratio: user.listed_count / (user.favourites_count + 1)
    user_listed_cols = [col for col in df_engineered.columns 
                       if 'listed_count' in col.lower() 
                       and 'quoted_status' not in col.lower()]
    user_favourites_cols = [col for col in df_engineered.columns 
                           if 'favourites_count' in col.lower() 
                           and 'quoted_status' not in col.lower()]
    
    if user_listed_cols and user_favourites_cols:
        listed_col = user_listed_cols[0]
        favourites_col = user_favourites_cols[0]
        
        df_engineered['listed_to_active_ratio'] = (
            df_engineered[listed_col].fillna(0) / 
            (df_engineered[favourites_col].fillna(0) + 1)
        )
        logger.info(f"  ✓ listed_to_active_ratio")
    
    # Tweet-level Features
    # is_reply is now handled in transform_extract_binary_flag above
    # But we also check in_reply_to_user_id for target encoding later
    if 'is_reply' not in df_engineered.columns:
        # Fallback: try to extract from in_reply_to_status_id if not already processed
        reply_cols = ['in_reply_to_status_id', 'in_reply_to_status_id_str']
        for col in reply_cols:
            if col in df_engineered.columns:
                df_engineered['is_reply'] = df_engineered[col].notna().astype(int)
                logger.info(f"  ✓ is_reply (fallback)")
                break
    
    # ========== Linguistic Behavior Patterns ==========
    logger.info("\n[17] Creating linguistic behavior pattern features...")
    
    # Mention count: from entities or text metadata
    if 'entities.user_mentions' in df_engineered.columns:
        df_engineered['mentions_count'] = df_engineered['entities.user_mentions'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        logger.info(f"  ✓ mentions_count (from entities)")
    else:
        # Try to get from text metadata if available
        text_mention_cols = [col for col in df_engineered.columns if 'mention_count' in col.lower()]
        if text_mention_cols:
            df_engineered['mentions_count'] = df_engineered[text_mention_cols[0]].fillna(0)
            logger.info(f"  ✓ mentions_count (from text metadata)")
    
    # Hashtag count: from entities or text metadata
    if 'entities.hashtags' in df_engineered.columns:
        df_engineered['hashtags_count'] = df_engineered['entities.hashtags'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else np.nan)
        )
        logger.info(f"  ✓ hashtags_count (from entities)")
    else:
        # Try to get from text metadata if available
        text_hashtag_cols = [col for col in df_engineered.columns if 'hashtag_count' in col.lower()]
        if text_hashtag_cols:
            df_engineered['hashtags_count'] = df_engineered[text_hashtag_cols[0]].fillna(0)
            logger.info(f"  ✓ hashtags_count (from text metadata)")
    
    # URL presence: from entities or text metadata
    if 'entities.urls' in df_engineered.columns:
        df_engineered['has_url'] = (df_engineered['entities.urls'].apply(
            lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else 0)
        ) > 0).astype(int)
        logger.info(f"  ✓ has_url (from entities)")
    else:
        # Try to get from text metadata if available
        text_url_cols = [col for col in df_engineered.columns if 'has_url' in col.lower() and 'text' in col.lower()]
        if text_url_cols:
            df_engineered['has_url'] = df_engineered[text_url_cols[0]].fillna(0)
            logger.info(f"  ✓ has_url (from text metadata)")
    
    # Broadcast intensity: mention_count (already created above)
    # Link density: has_url (already created above)
    # Hashtag complexity: hashtags_count (already created above)
    
    # ========== Temporal & Habitual Fingerprints ==========
    logger.info("\n[18] Creating temporal and habitual fingerprint features...")
    
    # is_business_hours: Extract hour from created_at and categorize
    created_at_hour_cols = [col for col in df_engineered.columns if 'created_at_hour' in col.lower()]
    if created_at_hour_cols:
        hour_col = created_at_hour_cols[0]
        
        # Categorize hours: Work (9-17), Evening (18-23), Sleep (0-8)
        def categorize_hour(hour):
            if pd.isna(hour):
                return None
            hour_int = int(hour) if not pd.isna(hour) else None
            if hour_int is None:
                return None
            if 9 <= hour_int <= 17:
                return 'work'
            elif 18 <= hour_int <= 23:
                return 'evening'
            else:  # 0-8
                return 'sleep'
        
        df_engineered['hour_category'] = df_engineered[hour_col].apply(categorize_hour)
        
        # One-hot encode hour category
        if is_train:
            # Get unique categories from training data
            unique_cats = df_engineered['hour_category'].dropna().unique()
            transformers['hour_category_cats'] = unique_cats
        else:
            unique_cats = transformers.get('hour_category_cats', ['work', 'evening', 'sleep'])
        
        for cat in ['work', 'evening', 'sleep']:
            df_engineered[f'is_{cat}_hours'] = (df_engineered['hour_category'] == cat).astype(int)
        
        # Drop intermediate column
        df_engineered = df_engineered.drop(columns=['hour_category'])
        logger.info(f"  ✓ is_business_hours (work/evening/sleep)")
    
    # post_regularity: Calculate variance of posting times (if user_id available and multiple tweets per user)
    # Note: This requires grouping by user_id, which may not be available in single-tweet prediction scenario
    # We'll skip this for now as it requires historical data per user
    
    # Drop entity list columns after extracting counts
    entity_drop_cols = [
        'entities.user_mentions', 'entities.hashtags', 'entities.symbols', 'entities.urls',
        'entities.media', 'extended_entities.media',
        'extended_tweet.entities.user_mentions', 'extended_tweet.entities.hashtags',
        'extended_tweet.entities.symbols', 'extended_tweet.entities.urls',
        'extended_tweet.entities.media', 'extended_tweet.extended_entities.media',
        'quoted_status.entities.user_mentions', 'quoted_status.entities.hashtags',
        'quoted_status.entities.symbols', 'quoted_status.entities.urls',
        'quoted_status.entities.media', 'quoted_status.extended_entities.media',
        'quoted_status.extended_tweet.entities.user_mentions',
        'quoted_status.extended_tweet.entities.hashtags',
        'quoted_status.extended_tweet.entities.symbols',
        'quoted_status.extended_tweet.entities.urls',
        'quoted_status.extended_tweet.entities.media',
        'quoted_status.extended_tweet.extended_entities.media',
    ]
    for col in entity_drop_cols:
        if col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[col])
    
    # ========== Text Embedding Processing ==========
    if use_text_embedding and TEXT_EMBEDDING_AVAILABLE and text_cols_for_embedding:
        logger.info(f"\n[19] Processing text embeddings for {len(text_cols_for_embedding)} text columns...")
        try:
            df_engineered, transformers = process_text_embeddings(
                df_engineered,
                text_cols_for_embedding,
                is_train=is_train,
                transformers=transformers,
                n_components=text_embedding_n_components
            )
            logger.info(f"  ✓ Text embedding processing complete")
        except Exception as e:
            logger.warning(f"  ⚠ Text embedding failed: {e}. Continuing without text embeddings.")
    elif use_text_embedding and not TEXT_EMBEDDING_AVAILABLE:
        logger.info(f"\n[19] Text embedding requested but transformers library not available. Skipping.")
    
    # Ensure all columns are numeric
    final_non_numeric = df_engineered.select_dtypes(exclude=[np.number]).columns.tolist()
    if final_non_numeric:
        logger.warning(f"  ⚠ {len(final_non_numeric)} non-numeric columns remain, dropping them")
        df_engineered = df_engineered.drop(columns=final_non_numeric)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Feature Engineering Complete!")
    logger.info(f"  Original columns: {len(df.columns)}")
    logger.info(f"  Final numeric columns: {len(df_engineered.columns)}")
    logger.info(f"  Processed: Drop={processed_count['drop']}, Keep={processed_count['keep']}, Transform={processed_count['transform']}")
    logger.info(f"{'='*60}")
    
    return df_engineered, transformers

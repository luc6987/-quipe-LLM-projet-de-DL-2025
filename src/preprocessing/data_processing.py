"""
Data Processing and Feature Engineering Utilities


Consolidates data loading, feature engineering, text/non-text splitting, and semantic integration.
/
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

# Import path configuration
from src.utils.paths import get_data_cache_dir

logger = logging.getLogger(__name__)

# Constants (Fallbacks only, should come from config)
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
CLEAN_CACHE_SUFFIX = "_cleaned.pkl"


# =============================================================================
# 1. Data Loading & Normalization
# =============================================================================

def _build_cache_path(file_path: str, cache_dir: Optional[Union[str, Path]]) -> Path:
    if cache_dir:
        base_dir = Path(cache_dir)
    else:
        base_dir = get_data_cache_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    return base_dir / f"{stem}{CLEAN_CACHE_SUFFIX}"

def normalize_tweet_text(text: Union[str, float, int]) -> str:
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

def normalize_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return df
    
    logger.info(f"Normalizing text columns: {columns}")
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
    Load data from JSONL. Optionally clean text columns.
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
            if i % 50000 == 0 and i > 0:
                logger.info(f"  Processed {i} lines...")
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"  Loaded {len(records)} records, normalizing JSON...")
    df = json_normalize(records)
    
    if clean_text:
        # If columns not provided, will be handled by config in later steps or default
        if text_columns:
             df = normalize_text_columns(df, columns=text_columns)
        
        if cache_path:
            df.to_pickle(cache_path)
    
    return df


# =============================================================================
# 2. Feature Split & Semantic Merger
# =============================================================================

def split_text_nontext(df: pd.DataFrame, text_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into Text DataFrame (for embeddings) and Non-Text DataFrame (numeric/categorical).
    """
    logger.info("Splitting Text and Non-Text features...")
    
    existing_text_cols = [c for c in text_columns if c in df.columns]
    
    text_df = df[existing_text_cols].copy() if existing_text_cols else pd.DataFrame(index=df.index)
    non_text_df = df.drop(columns=existing_text_cols).copy()
    
    return text_df, non_text_df

def merge_semantic_features(
    df: pd.DataFrame, 
    embedding_path: Union[str, Path],
    embedding_dim: int = 768
) -> pd.DataFrame:
    """
    Merge pre-computed semantic embeddings into the DataFrame.
    """
    emb_path = Path(embedding_path)
    if not emb_path.exists():
        logger.warning(f"Embedding file not found at {emb_path}. Skipping semantic merge.")
        return df
        
    logger.info(f"Loading embeddings from {emb_path}...")
    try:
        embeddings = np.load(emb_path)
        
        if len(embeddings) != len(df):
            logger.error(f"Embedding shape {embeddings.shape} mismatch with DataFrame {len(df)}")
            return df
            
        if embeddings.shape[1] != embedding_dim:
             logger.warning(f"Embedding dim {embeddings.shape[1]} != configured {embedding_dim}")
             
        # Create DataFrame from embeddings
        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
        
        # Concatenate
        df_merged = pd.concat([df, emb_df], axis=1)
        logger.info(f"Merged semantic features. New shape: {df_merged.shape}")
        return df_merged
        
    except Exception as e:
        logger.error(f"Failed to load/merge embeddings: {e}")
        return df


# =============================================================================
# 3. Feature Engineering (Config Driven)
# =============================================================================

def calculate_account_age(created_at_str: str) -> Optional[int]:
    if pd.isna(created_at_str) or created_at_str is None:
        return None
    try:
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        now = datetime.now(created_at.tzinfo)
        return max(0, (now - created_at).days)
    except (ValueError, TypeError):
        return None

def filter_features_by_shap(
    df: pd.DataFrame,
    shap_json_path: Union[str, Path],
    threshold: float = 0.005
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter features based on SHAP importance values.
    SHAP
    
    Args:
        df: DataFrame with features to filter
        shap_json_path: Path to feature_importance.json file
        threshold: Minimum SHAP value to keep (features below this will be dropped)
        
    Returns:
        Tuple of (filtered DataFrame, list of dropped feature names)
    """
    shap_json_path = Path(shap_json_path)
    
    if not shap_json_path.exists():
        logger.debug(f"SHAP JSON file not found at {shap_json_path}, skipping feature filtering")
        return df, []
    
    try:
        with open(shap_json_path, 'r', encoding='utf-8') as f:
            shap_dict = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load SHAP JSON from {shap_json_path}: {e}. Skipping feature filtering.")
        return df, []
    
    # Find features to drop (SHAP < threshold)
    features_to_drop = []
    for feature_name, shap_value in shap_dict.items():
        if shap_value < threshold:
            if feature_name in df.columns:
                features_to_drop.append(feature_name)
    
    if features_to_drop:
        logger.info(f"[SHAP Filter] Dropping {len(features_to_drop)} features with SHAP < {threshold}")
        logger.info(f"  Dropped features: {', '.join(features_to_drop[:10])}" + 
                   (f" ... and {len(features_to_drop) - 10} more" if len(features_to_drop) > 10 else ""))
        df_filtered = df.drop(columns=features_to_drop, errors='ignore')
        return df_filtered, features_to_drop
    else:
        logger.info(f"[SHAP Filter] No features below threshold {threshold} to drop")
        return df, []

def engineer_features(
    df: pd.DataFrame,
    config: Dict,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    hashtag_lib: Optional[Dict] = None,
    mention_lib: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Engineer features based on Configuration.
    Strictly follows setup.yml parameters.
    """
    logger.info("Starting Feature Engineering...")
    
    df_eng = df.copy()
    if transformers is None:
        transformers = {}
        
    prep_config = config.get('preprocessing', {})
    
    # --- 1. Library Features (Hashtag/Mention) ---
    if hashtag_lib:
        logger.info("[1] Creating Hashtag Library Features")
        def check_hashtag_lib(text):
            if not isinstance(text, str): return 0
            # Simple check: count how many top hashtags appear
            count = 0
            text_lower = text.lower()
            for tag in hashtag_lib:
                if f"#{tag}" in text_lower:
                    count += 1
            return count
            
        # Apply to main text column
        if 'text' in df_eng.columns:
            df_eng['hashtag_lib_count'] = df_eng['text'].apply(check_hashtag_lib)
            
    if mention_lib:
        logger.info("[1] Creating Mention Library Features")
        def check_mention_lib(text):
            if not isinstance(text, str): return 0
            count = 0
            text_lower = text.lower()
            for user in mention_lib:
                if f"@{user}" in text_lower:
                    count += 1
            return count
            
        if 'text' in df_eng.columns:
            df_eng['mention_lib_count'] = df_eng['text'].apply(check_mention_lib)

    # --- 2. Numeric Transformations (Log Skewed) ---
    skewed_cols = prep_config.get('skewed_features', [])
    if skewed_cols:
        logger.info(f"[2] Log-transforming {len(skewed_cols)} skewed features")
        for col in skewed_cols:
             # Match suffix approach
             matching = [c for c in df_eng.columns if c.endswith(col) or c == col]
             for match in matching:
                 if pd.api.types.is_numeric_dtype(df_eng[match]):
                     df_eng[f"log_{match}"] = np.log1p(df_eng[match].fillna(0))

    # --- 3. Derived Features (Account Age, Ratios) ---
    logger.info("[3] Creating derived features (Account Age, Ratios)")
    # Account Age
    user_created = [c for c in df_eng.columns if 'created_at' in c.lower() and 'user' in c.lower()]
    if user_created:
        df_eng['account_age_days'] = df_eng[user_created[0]].apply(calculate_account_age)
        
    # Follower/Following Ratio
    followers = [c for c in df_eng.columns if 'followers_count' in c.lower() and 'quoted' not in c.lower()]
    friends = [c for c in df_eng.columns if 'friends_count' in c.lower() and 'quoted' not in c.lower()]
    
    if followers and friends:
        f_col, fr_col = followers[0], friends[0]
        epsilon = 1e-6
        df_eng['follower_following_ratio'] = df_eng[f_col] / (df_eng[fr_col] + epsilon)

    # --- 4. Official/Verified Word Check (from Description) ---
    official_words = prep_config.get('official_words', [])
    if official_words and 'user.description' in df_eng.columns:
        logger.info("[4] Checking Official Words in Description")
        pattern = '|'.join([f'\\b{word}\\b' for word in official_words])
        df_eng['has_official_word'] = df_eng['user.description'].astype(str).str.contains(
            pattern, case=False, na=False, regex=True).astype(int)

    # --- 5. Encode Categoricals ---
    logger.info("[5] Encoding Categoricals")
    # Identify object columns excluding text/id/label
    text_cols = prep_config.get('text_columns', [])
    drop_cols = prep_config.get('drop_columns', [])
    
    # Exclude text columns and explicitly dropped columns from encoding
    cols_to_exclude = set(text_cols + drop_cols + ['challenge_id', 'label', 'id', 'id_str'])
    
    cat_cols = [c for c in df_eng.select_dtypes(include=['object', 'string']).columns 
                if c not in cols_to_exclude]
    
    if cat_cols:
        cat_data = df_eng[cat_cols].fillna('unknown').astype(str).values
        
        if is_train:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            enc.fit(cat_data)
            transformers['ord_encoder'] = enc
            transformers['ord_cols'] = cat_cols
        else:
            # Align cols
            train_cols = transformers.get('ord_cols', [])
            # Add missing
            for c in train_cols:
                if c not in df_eng.columns: df_eng[c] = 'unknown'
            # Drop extra
            df_eng = df_eng.drop(columns=[c for c in cat_cols if c not in train_cols])
            cat_cols = train_cols
            
            # Encode
            cat_data = df_eng[cat_cols].fillna('unknown').astype(str).values
            enc = transformers.get('ord_encoder')
            if enc is None:
                 # Fallback
                 enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                 enc.fit(cat_data)

        encoded = enc.transform(cat_data)
        for i, col in enumerate(cat_cols):
            df_eng[col] = encoded[:, i]

    # --- 6. Drop Columns ---
    logger.info("[6] Dropping specified columns")
    cols_to_drop = [c for c in drop_cols if c in df_eng.columns]
    
    # Also drop text columns if split_text_nontext is True (handled by caller, but good to clean)
    if prep_config.get('split_text_nontext', True):
        cols_to_drop.extend([c for c in text_cols if c in df_eng.columns])
        
    df_eng.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')
    
    # Clean non-numerics
    non_nums = df_eng.select_dtypes(exclude=[np.number]).columns.tolist()
    # Keep challenge_id
    if 'challenge_id' in non_nums: non_nums.remove('challenge_id')
    if non_nums:
        logger.warning(f"Dropping residual non-numeric columns: {non_nums}")
        df_eng.drop(columns=non_nums, inplace=True)
    
    # --- 7. SHAP-based Feature Filtering (if enabled and SHAP JSON exists) ---
    opt_config = config.get('optimization', {})
    use_shap_filter = opt_config.get('use_shap_filter', False)
    shap_filter_threshold = opt_config.get('shap_filter_threshold', 0.005)
    
    if use_shap_filter and is_train:
        # Get SHAP directory from config
        paths_config = config.get('paths', {})
        shap_dir = Path(paths_config.get('shap_dir', 'output/shap'))
        shap_json_path = shap_dir / 'feature_importance.json'
        
        if shap_json_path.exists():
            logger.info(f"[7] Applying SHAP-based feature filtering (threshold={shap_filter_threshold})...")
            df_eng, dropped_features = filter_features_by_shap(
                df_eng, 
                shap_json_path, 
                threshold=shap_filter_threshold
            )
            if dropped_features:
                logger.info(f"  ✓ Filtered out {len(dropped_features)} low-importance features")
        else:
            logger.debug(f"[7] SHAP JSON not found at {shap_json_path}, skipping SHAP filtering (first run?)")
    elif use_shap_filter and not is_train:
        # For test set, also apply filtering if enabled
        paths_config = config.get('paths', {})
        shap_dir = Path(paths_config.get('shap_dir', 'output/shap'))
        shap_json_path = shap_dir / 'feature_importance.json'
        
        if shap_json_path.exists():
            logger.info(f"[7] Applying SHAP-based feature filtering to test set (threshold={shap_filter_threshold})...")
            df_eng, dropped_features = filter_features_by_shap(
                df_eng, 
                shap_json_path, 
                threshold=shap_filter_threshold
            )
            if dropped_features:
                logger.info(f"  ✓ Filtered out {len(dropped_features)} low-importance features from test set")
        
    return df_eng, transformers

def prepare_xy(df: pd.DataFrame, target_col: str = 'label', id_col: str = 'challenge_id'):
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
    
    ids = None
    if id_col in X.columns:
        ids = X[id_col]
        X = X.drop(columns=[id_col])
        
    return X, y, ids

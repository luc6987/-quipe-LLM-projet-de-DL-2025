"""
Text Preprocessing Module for CamemBERT
CamemBERT 文本预处理模块

Handles text field fusion, style feature extraction, URL classification, and normalization.
"""

import re
import html
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
from src.utils.paths import get_preprocessing_dir

logger = logging.getLogger(__name__)

# URL patterns for classification
MEDIA_URL_PATTERNS = [
    r'pic\.twitter\.com',
    r'pbs\.twimg\.com',
    r'twimg\.com',
    r'media\.twitter\.com',
]

LINK_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MEDIA_URL_PATTERN = re.compile('|'.join(MEDIA_URL_PATTERNS), re.IGNORECASE)
USERNAME_PATTERN = re.compile(r'@[A-Za-z0-9_]{1,30}')
MULTISPACE_PATTERN = re.compile(r'\s+')


def extract_and_fuse_text_fields(df: pd.DataFrame) -> pd.Series:
    """
    Extract and fuse text fields from tweet data.
    提取并融合推文数据中的文本字段。
    
    Strategy:
    - Priority: extended_tweet.full_text > text
    - If quoted_status.text exists: text [SEP] extended_tweet.full_text [SEP] [QUOTED] quoted_status.text
    - If user.description exists: append [USER_DESC] user.description
    - Handle missing values with empty strings
    
    Args:
        df: DataFrame with nested JSON columns
        
    Returns:
        Series of fused text strings
    """
    logger.info("Extracting and fusing text fields...")
    
    # Initialize with empty strings
    fused_texts = pd.Series([''] * len(df), index=df.index)
    
    # Extract base text fields
    text_col = 'text' if 'text' in df.columns else None
    extended_text_col = 'extended_tweet.full_text' if 'extended_tweet.full_text' in df.columns else None
    quoted_text_col = 'quoted_status.text' if 'quoted_status.text' in df.columns else None
    user_desc_col = 'user.description' if 'user.description' in df.columns else None
    
    for idx in df.index:
        parts = []
        
        # Get main text (prefer extended_tweet.full_text over text)
        main_text = ""
        if extended_text_col and pd.notna(df.loc[idx, extended_text_col]):
            main_text = str(df.loc[idx, extended_text_col])
        elif text_col and pd.notna(df.loc[idx, text_col]):
            main_text = str(df.loc[idx, text_col])
        
        if main_text:
            parts.append(main_text)
        
        # Add quoted text if exists
        if quoted_text_col and pd.notna(df.loc[idx, quoted_text_col]):
            quoted_text = str(df.loc[idx, quoted_text_col])
            if quoted_text:
                parts.append(f"[QUOTED] {quoted_text}")
        
        # Add user description if exists
        if user_desc_col and pd.notna(df.loc[idx, user_desc_col]):
            user_desc = str(df.loc[idx, user_desc_col])
            if user_desc:
                parts.append(f"[USER_DESC] {user_desc}")
        
        # Join with [SEP] separator
        fused_text = " [SEP] ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        fused_texts.loc[idx] = fused_text if fused_text else ""
    
    logger.info(f"Fused text fields: {len(fused_texts)} texts, avg length: {fused_texts.str.len().mean():.1f}")
    return fused_texts


def extract_text_style_features(texts: pd.Series) -> pd.DataFrame:
    """
    Extract text style features BEFORE cleaning to preserve style signals.
    在清理前提取文本风格特征以保留风格信号。
    
    Features (~25 dimensions):
    - Basic stats: text_length, word_count, avg_word_length
    - Case: uppercase_ratio, has_all_uppercase_words
    - Punctuation: exclamation_count, question_count, ratios, has_repeated_punctuation
    - Character repetition: has_repeated_chars, repeated_char_ratio
    - Special markers: has_rt_marker, has_ellipsis, has_dash
    - Format: newline_count, tab_count, consecutive_space_count
    
    Args:
        texts: Series of text strings (raw, before cleaning)
        
    Returns:
        DataFrame with style feature columns (prefix: text_style_*)
    """
    logger.info("Extracting text style features (before cleaning)...")
    
    features = pd.DataFrame(index=texts.index)
    
    # Fill NaN with empty strings
    texts = texts.fillna("").astype(str)
    
    # Basic statistics
    features['text_style_length'] = texts.str.len()
    features['text_style_word_count'] = texts.str.split().str.len()
    features['text_style_avg_word_length'] = (
        features['text_style_length'] / features['text_style_word_count'].replace(0, 1)
    )
    
    # Case features
    features['text_style_uppercase_count'] = texts.str.findall(r'[A-Z]').str.len()
    features['text_style_uppercase_ratio'] = (
        features['text_style_uppercase_count'] / features['text_style_length'].replace(0, 1)
    )
    features['text_style_has_all_uppercase_words'] = texts.str.contains(r'\b[A-Z]{2,}\b').astype(int)
    
    # Punctuation features
    features['text_style_exclamation_count'] = texts.str.count('!')
    features['text_style_question_count'] = texts.str.count(r'\?')
    features['text_style_exclamation_ratio'] = (
        features['text_style_exclamation_count'] / features['text_style_length'].replace(0, 1)
    )
    features['text_style_question_ratio'] = (
        features['text_style_question_count'] / features['text_style_length'].replace(0, 1)
    )
    features['text_style_has_repeated_punctuation'] = texts.str.contains(r'[!?]{2,}').astype(int)
    
    # Character repetition (e.g., "sooo", "yesss")
    features['text_style_has_repeated_chars'] = texts.str.contains(r'(.)\1{2,}').astype(int)
    # Count repeated character patterns
    repeated_char_pattern = re.compile(r'(.)\1{2,}')
    features['text_style_repeated_char_ratio'] = texts.apply(
        lambda x: len(repeated_char_pattern.findall(x)) / max(len(x), 1)
    )
    
    # Special markers
    features['text_style_has_rt_marker'] = texts.str.contains(r'\bRT\b', case=False).astype(int)
    features['text_style_has_ellipsis'] = texts.str.contains(r'\.{3,}').astype(int)
    features['text_style_has_dash'] = texts.str.contains(r'[-—–]').astype(int)
    
    # Format features
    features['text_style_newline_count'] = texts.str.count('\n')
    features['text_style_tab_count'] = texts.str.count('\t')
    features['text_style_consecutive_space_count'] = texts.str.count(r' {2,}')
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    logger.info(f"Extracted {len(features.columns)} style features")
    return features


def classify_and_replace_urls(
    df: pd.DataFrame,
    texts: pd.Series
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Classify URLs into media vs link URLs and replace with type markers.
    将URL分类为媒体URL和链接URL，并用类型标记替换。
    
    Classification:
    - Media URLs: From entities.media, extended_entities.media, or patterns (pic.twitter.com, pbs.twimg.com)
    - Link URLs: From entities.urls or general http/https patterns
    
    Replacement:
    - Media URLs → <MEDIA_URL>
    - Link URLs → <LINK_URL>
    
    Additional Features:
    - has_media_url, has_link_url, media_url_count, link_url_count
    
    Args:
        df: DataFrame with entities columns
        texts: Series of text strings
        
    Returns:
        Tuple of (cleaned_texts with URL markers, URL feature DataFrame)
    """
    logger.info("Classifying and replacing URLs...")
    
    cleaned_texts = texts.copy()
    url_features = pd.DataFrame(index=texts.index)
    
    # Initialize URL feature columns
    url_features['has_media_url'] = 0
    url_features['has_link_url'] = 0
    url_features['media_url_count'] = 0
    url_features['link_url_count'] = 0
    
    # Extract media URLs from entities
    media_urls_set = set()
    for col in ['entities.media', 'extended_entities.media', 'extended_tweet.entities.media']:
        if col in df.columns:
            for idx in df.index:
                try:
                    media_data = df.loc[idx, col]
                    # Check if value is not null/NaN - handle both scalar and array cases
                    if media_data is not None and (not isinstance(media_data, float) or not pd.isna(media_data)):
                        if isinstance(media_data, list):
                            for item in media_data:
                                if isinstance(item, dict) and 'media_url' in item:
                                    media_urls_set.add(item['media_url'])
                                elif isinstance(item, dict) and 'url' in item:
                                    media_urls_set.add(item['url'])
                except:
                    pass
    
    # Extract link URLs from entities
    link_urls_set = set()
    for col in ['entities.urls', 'extended_tweet.entities.urls', 'quoted_status.entities.urls']:
        if col in df.columns:
            for idx in df.index:
                try:
                    url_data = df.loc[idx, col]
                    # Check if value is not null/NaN - handle both scalar and array cases
                    if url_data is not None and (not isinstance(url_data, float) or not pd.isna(url_data)):
                        if isinstance(url_data, list):
                            for item in url_data:
                                if isinstance(item, dict) and 'url' in item:
                                    link_urls_set.add(item['url'])
                                elif isinstance(item, dict) and 'expanded_url' in item:
                                    link_urls_set.add(item['expanded_url'])
                except:
                    pass
    
    # Process each text
    for idx in texts.index:
        text = str(texts.loc[idx]) if pd.notna(texts.loc[idx]) else ""
        
        # Find all URLs in text
        all_urls = LINK_URL_PATTERN.findall(text)
        
        media_count = 0
        link_count = 0
        
        for url in all_urls:
            # Check if it's a media URL (from entities or pattern)
            is_media = (
                any(media_url in url for media_url in media_urls_set) or
                MEDIA_URL_PATTERN.search(url) is not None
            )
            
            if is_media:
                text = text.replace(url, '<MEDIA_URL>', 1)
                media_count += 1
            else:
                text = text.replace(url, '<LINK_URL>', 1)
                link_count += 1
        
        cleaned_texts.loc[idx] = text
        
        # Update URL features
        url_features.loc[idx, 'has_media_url'] = 1 if media_count > 0 else 0
        url_features.loc[idx, 'has_link_url'] = 1 if link_count > 0 else 0
        url_features.loc[idx, 'media_url_count'] = media_count
        url_features.loc[idx, 'link_url_count'] = link_count
    
    logger.info(f"URL classification complete: {url_features['has_media_url'].sum()} media URLs, "
                f"{url_features['has_link_url'].sum()} link URLs")
    
    return cleaned_texts, url_features


def normalize_text_for_camembert(texts: pd.Series) -> pd.Series:
    """
    Normalize text for CamemBERT with minimal cleaning (preserve style).
    为CamemBERT规范化文本，最小化清理（保留风格）。
    
    Cleaning steps:
    - HTML unescape
    - Unicode normalization (NFKC)
    - Replace URLs with type markers (already done in classify_and_replace_urls)
    - Replace usernames with @USER
    - Normalize excessive whitespace (keep single spaces)
    
    Preserve:
    - Emoji (keep original)
    - Case differences
    - Punctuation repetition
    - Character repetition
    
    Args:
        texts: Series of text strings (after URL replacement)
        
    Returns:
        Series of normalized text strings
    """
    logger.info("Normalizing text for CamemBERT (preserving style)...")
    
    normalized = texts.copy()
    
    for idx in texts.index:
        text = str(texts.loc[idx]) if pd.notna(texts.loc[idx]) else ""
        
        # HTML unescape
        text = html.unescape(text)
        
        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace usernames with @USER
        text = USERNAME_PATTERN.sub('@USER', text)
        
        # Normalize excessive whitespace (keep single spaces, preserve newlines)
        # First, replace newlines with a placeholder
        text = text.replace('\n', '\n')
        text = MULTISPACE_PATTERN.sub(' ', text)
        text = text.strip()
        
        normalized.loc[idx] = text
    
    logger.info(f"Text normalization complete: {len(normalized)} texts processed")
    return normalized


def load_preprocessed_camembert_texts(
    preprocess_dir: Optional[Path] = None
) -> Optional[Dict[str, pd.Series]]:
    """
    Load preprocessed texts and labels saved during unified preprocessing.
    从统一预处理中加载已处理的文本与标签。
    """
    base = Path(preprocess_dir) if preprocess_dir else get_preprocessing_dir()
    train_text_path = base / "camembert_train_texts.pkl"
    test_text_path = base / "camembert_test_texts.pkl"
    label_path = base / "train_labels.pkl"
    test_ids_path = base / "test_ids.pkl"

    if not (train_text_path.exists() and test_text_path.exists() and label_path.exists()):
        logger.warning("Preprocessed CamemBERT texts not found in %s", base)
        return None

    try:
        train_texts = pd.read_pickle(train_text_path)
        test_texts = pd.read_pickle(test_text_path)
        train_labels = pd.read_pickle(label_path)
        test_ids = pd.read_pickle(test_ids_path) if test_ids_path.exists() else None
        return {
            "train_texts": train_texts,
            "test_texts": test_texts,
            "train_labels": train_labels,
            "test_ids": test_ids,
        }
    except Exception as exc:
        logger.warning("Failed to load preprocessed CamemBERT texts: %s", exc)
        return None


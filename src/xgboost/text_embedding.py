"""
Text Embedding Module
文本嵌入模块 / Module d'embedding de texte / Text Embedding Module / Text-Embedding-Modul

Extracts DeBERTa embeddings and applies PCA for dimensionality reduction.
提取DeBERTa嵌入并应用PCA进行降维。
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
# 从统一导入模块导入 PCA
from src._imports import sklearn_PCA as PCA
import logging
import os

logger = logging.getLogger(__name__)

# Use unified imports module
from src._imports import (
    TRANSFORMERS_AVAILABLE, AutoModel, DebertaV2Tokenizer, torch
)


def load_deberta_model(model_name: str = "microsoft/deberta-v3-base"):
    """
    Load DeBERTa model and tokenizer.
    加载DeBERTa模型和分词器。
    
    Args:
        model_name: Name of the model to load (default: microsoft/deberta-v3-base)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for text embedding")
    
    logger.info(f"Loading DeBERTa model: {model_name}")
    # DeBERTa v3 requires DebertaV2Tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"Model loaded on device: {device}")
    return tokenizer, model, device


def extract_deberta_embeddings(
    texts: pd.Series,
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 128
) -> np.ndarray:
    """
    Extract DeBERTa embeddings from text series.
    从文本序列提取DeBERTa嵌入（最后一层的[CLS] token）。
    
    Args:
        texts: Series of text strings
        tokenizer: DeBERTa tokenizer
        model: DeBERTa model
        device: torch device (cpu or cuda)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        numpy array of shape (n_samples, 768) containing [CLS] token embeddings from last hidden state
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for text embedding")
    
    embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts.iloc[i:i+batch_size].fillna("").astype(str).tolist()
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings (no gradient computation)
        with torch.no_grad():
            outputs = model(**encoded)
            # Extract [CLS] token embedding from last hidden state (first token of last layer)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch_texts)} / {len(texts)} texts...")
    
    # Concatenate all batches
    all_embeddings = np.vstack(embeddings)
    logger.info(f"Extracted embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings


def apply_pca_reduction(
    embeddings: np.ndarray,
    n_components: int = 32,
    is_train: bool = True,
    transformers: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply PCA for dimensionality reduction.
    应用PCA进行降维。
    
    Args:
        embeddings: Input embeddings (n_samples, 768)
        n_components: Number of PCA components (default: 32)
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        
    Returns:
        Tuple of (reduced_embeddings, updated transformers dict)
    """
    if transformers is None:
        transformers = {}
    
    transformer_key = 'pca_text_embedding'
    
    if is_train:
        # Fit PCA on training data
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        transformers[transformer_key] = pca
        
        # Log explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    else:
        # Transform using fitted PCA
        pca = transformers.get(transformer_key)
        if pca is None:
            raise ValueError("PCA transformer not found. Must fit on training data first.")
        reduced_embeddings = pca.transform(embeddings)
    
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings, transformers


def process_text_embeddings(
    df: pd.DataFrame,
    text_cols: list,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    model_name: str = "microsoft/deberta-v3-base",
    n_components: int = 32,
    batch_size: int = 32,
    max_length: int = 128
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process text columns to extract DeBERTa embeddings and apply PCA.
    处理文本列以提取DeBERTa嵌入并应用PCA。
    
    Args:
        df: Input DataFrame
        text_cols: List of text column names to process
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        model_name: DeBERTa model name (default: microsoft/deberta-v3-base)
        n_components: Number of PCA components (default: 32)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (DataFrame with embedding features, updated transformers dict)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers library not available. Skipping text embedding.")
        return df, transformers or {}
    
    if transformers is None:
        transformers = {}
    
    # Find available text columns
    available_cols = [col for col in text_cols if col in df.columns]
    
    if not available_cols:
        logger.info("No text columns found for embedding extraction.")
        return df, transformers
    
    logger.info(f"Processing text embeddings for {len(available_cols)} columns...")
    
    # Load model (only once)
    transformer_key_model = 'deberta_model_loaded'
    if transformer_key_model not in transformers:
        tokenizer, model, device = load_deberta_model(model_name)
        transformers[transformer_key_model] = {
            'tokenizer': tokenizer,
            'model': model,
            'device': device
        }
    else:
        model_info = transformers[transformer_key_model]
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        device = model_info['device']
    
    # Process each text column
    all_embeddings_list = []
    embedding_col_names = []
    
    for col in available_cols:
        logger.info(f"  Processing {col}...")
        texts = df[col]
        
        # Extract embeddings
        embeddings = extract_deberta_embeddings(
            texts, tokenizer, model, device, batch_size, max_length
        )
        
        all_embeddings_list.append(embeddings)
        embedding_col_names.append(col)
    
    # Combine embeddings from all text columns (average if multiple)
    if len(all_embeddings_list) == 1:
        combined_embeddings = all_embeddings_list[0]
    else:
        # Average embeddings from multiple text columns
        combined_embeddings = np.mean(all_embeddings_list, axis=0)
        logger.info("  Averaged embeddings from multiple text columns")
    
    # Apply PCA reduction
    reduced_embeddings, transformers = apply_pca_reduction(
        combined_embeddings, n_components, is_train, transformers
    )
    
    # Add embedding features to DataFrame
    # Keep column names as 'bert_*' for backward compatibility
    for i in range(n_components):
        col_name = f'bert_{i}'
        df[col_name] = reduced_embeddings[:, i]
    
    # Drop original text columns (metadata already extracted in feature_engineering)
    for col in available_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    logger.info(f"  ✓ Added {n_components} DeBERTa embedding features (bert_0 to bert_{n_components-1})")
    
    return df, transformers


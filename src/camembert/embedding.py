"""
CamemBERT Embedding Extraction Module
CamemBERT 嵌入提取模块

Handles CamemBERT model loading, embedding extraction, and PCA dimensionality reduction.
"""

# 从统一导入模块导入所有库
from .._imports import (
    np, pd, Path, Dict, Tuple, Optional, logging,
    TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, torch,
    CamembertTokenizer, CamembertModel, sklearn_PCA
)

# Use PCA from unified imports, with fallback
if sklearn_PCA is not None:
    PCA = sklearn_PCA
else:
    # Fallback if sklearn is not available
    from sklearn.decomposition import PCA

from ..preprocessing.camembert_preprocessing import (
    extract_and_fuse_text_fields,
    extract_text_style_features,
    classify_and_replace_urls,
    normalize_text_for_camembert
)

logger = logging.getLogger(__name__)


def load_camembert_model(model_name: str = "camembert-base"):
    """
    Load CamemBERT model and tokenizer.
    加载CamemBERT模型和分词器。
    
    Args:
        model_name: Name of the model to load (default: camembert-base)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for CamemBERT embedding")
    
    logger.info(f"Loading CamemBERT model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"CamemBERT model loaded on device: {device}")
    return tokenizer, model, device


def extract_camembert_embeddings(
    texts: pd.Series,
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 256
) -> np.ndarray:
    """
    Extract CamemBERT embeddings from text series.
    从文本序列提取CamemBERT嵌入（倒数第二层的[CLS] token）。
    
    Args:
        texts: Series of text strings
        tokenizer: CamemBERT tokenizer
        model: CamemBERT model
        device: torch device (cpu or cuda)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        numpy array of shape (n_samples, 768) containing [CLS] token embeddings from second-to-last hidden state
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for CamemBERT embedding")
    
    embeddings = []
    
    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts.iloc[i:i+batch_size].fillna("").astype(str).tolist()
        
        try:
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
                outputs = model(**encoded, output_hidden_states=True)
                # Extract [CLS] token embedding from second-to-last hidden state (倒数第二层)
                # outputs.hidden_states 是一个元组，[-2] 是倒数第二层，[-1] 是最后一层
                batch_embeddings = outputs.hidden_states[-2][:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
            
            # Progress logging
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"  Processed {min(i + len(batch_texts), len(texts))} / {len(texts)} texts...")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  GPU out of memory at batch {i // batch_size + 1}, reducing batch size...")
                # Try with smaller batch size
                if batch_size > 1:
                    # Recursively try with half batch size
                    smaller_batch = batch_size // 2
                    logger.info(f"  Retrying with batch size {smaller_batch}")
                    batch_embeddings = extract_camembert_embeddings(
                        texts.iloc[i:], tokenizer, model, device, smaller_batch, max_length
                    )
                    embeddings.append(batch_embeddings)
                else:
                    raise
            else:
                raise
    
    # Concatenate all batches
    if embeddings:
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Extracted CamemBERT embeddings shape: {all_embeddings.shape}")
        return all_embeddings
    else:
        raise ValueError("No embeddings extracted")


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
    
    transformer_key = 'pca_camembert_embedding'
    
    if is_train:
        # Fit PCA on training data
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
        transformers[transformer_key] = pca
        
        # Log explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
        
        if explained_var < 0.85:
            logger.warning(f"  PCA explained variance ({explained_var*100:.2f}%) is below recommended 85%")
    else:
        # Transform using fitted PCA
        pca = transformers.get(transformer_key)
        if pca is None:
            raise ValueError("PCA transformer not found. Must fit on training data first.")
        reduced_embeddings = pca.transform(embeddings)
    
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings, transformers


def process_camembert_embeddings(
    df: pd.DataFrame,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    model_name: str = "camembert-base",
    n_components: int = 32,
    batch_size: int = 32,
    max_length: int = 256,
    extract_style_features: bool = True,
    classify_urls: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process text to extract CamemBERT embeddings with preprocessing and PCA.
    处理文本以提取CamemBERT嵌入，包括预处理和PCA。
    
    This is the main entry point that integrates all preprocessing steps:
    1. Extract and fuse text fields
    2. Extract style features (before cleaning)
    3. Classify and replace URLs
    4. Normalize text
    5. Extract CamemBERT embeddings
    6. Apply PCA reduction
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        model_name: CamemBERT model name (default: camembert-base)
        n_components: Number of PCA components (default: 32)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        extract_style_features: Whether to extract style features
        classify_urls: Whether to classify and replace URLs
        
    Returns:
        Tuple of (DataFrame with embedding and style features, updated transformers dict)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers library not available. Skipping CamemBERT embedding.")
        return df, transformers or {}
    
    if transformers is None:
        transformers = {}
    
    logger.info("="*60)
    logger.info("Processing CamemBERT Embeddings")
    logger.info("="*60)
    
    # Step 1: Extract and fuse text fields
    logger.info("\n[Step 1] Extracting and fusing text fields...")
    fused_texts = extract_and_fuse_text_fields(df)
    
    # Step 2: Extract style features (BEFORE cleaning)
    style_features_df = pd.DataFrame(index=df.index)
    if extract_style_features:
        logger.info("\n[Step 2] Extracting text style features...")
        style_features_df = extract_text_style_features(fused_texts)
    
    # Step 3: Classify and replace URLs
    url_features_df = pd.DataFrame(index=df.index)
    if classify_urls:
        logger.info("\n[Step 3] Classifying and replacing URLs...")
        fused_texts, url_features_df = classify_and_replace_urls(df, fused_texts)
    else:
        # Still need to replace URLs with generic markers for normalization
        from ..preprocessing.camembert_preprocessing import LINK_URL_PATTERN
        fused_texts = fused_texts.str.replace(LINK_URL_PATTERN, '<LINK_URL>', regex=True)
    
    # Step 4: Normalize text
    logger.info("\n[Step 4] Normalizing text for CamemBERT...")
    normalized_texts = normalize_text_for_camembert(fused_texts)
    
    # Step 5: Load model (only once)
    transformer_key_model = 'camembert_model_loaded'
    if transformer_key_model not in transformers:
        logger.info("\n[Step 5] Loading CamemBERT model...")
        tokenizer, model, device = load_camembert_model(model_name)
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
        logger.info(f"\n[Step 5] Using cached CamemBERT model on {device}")
    
    # Step 6: Extract embeddings
    logger.info("\n[Step 6] Extracting CamemBERT embeddings...")
    embeddings = extract_camembert_embeddings(
        normalized_texts, tokenizer, model, device, batch_size, max_length
    )
    
    # Step 7: Apply PCA reduction
    logger.info("\n[Step 7] Applying PCA reduction...")
    reduced_embeddings, transformers = apply_pca_reduction(
        embeddings, n_components, is_train, transformers
    )
    
    # Step 8: Add embedding features to DataFrame
    logger.info("\n[Step 8] Adding features to DataFrame...")
    for i in range(n_components):
        col_name = f'camembert_{i}'
        df[col_name] = reduced_embeddings[:, i]
    
    # Step 9: Add style features
    if extract_style_features and not style_features_df.empty:
        for col in style_features_df.columns:
            df[col] = style_features_df[col]
    
    # Step 10: Add URL features
    if classify_urls and not url_features_df.empty:
        for col in url_features_df.columns:
            df[col] = url_features_df[col]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CamemBERT Processing Complete!")
    logger.info(f"  - Embedding features: {n_components} (camembert_0 to camembert_{n_components-1})")
    if extract_style_features:
        logger.info(f"  - Style features: {len(style_features_df.columns)}")
    if classify_urls:
        logger.info(f"  - URL features: {len(url_features_df.columns)}")
    logger.info(f"{'='*60}\n")
    
    return df, transformers


def process_finetuned_probability_features(
    df: pd.DataFrame,
    finetuned_model_path: Path,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    n_components: int = 32,
    batch_size: int = 32,
    max_length: int = 256,
    extract_style_features: bool = True,
    classify_urls: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process texts to extract probability features from fine-tuned model and apply PCA.
    处理文本以从微调后的模型提取概率特征并应用PCA。
    
    This function integrates:
    1. Probability extraction from fine-tuned model
    2. PCA dimensionality reduction (from 2D probabilities to n_components)
    3. Style and URL feature extraction
    
    Args:
        df: Input DataFrame
        finetuned_model_path: Path to fine-tuned CamemBERT model
        is_train: Whether this is training data
        transformers: Dictionary of transformers
        n_components: Number of PCA components (default: 32)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        extract_style_features: Whether to extract style features
        classify_urls: Whether to classify and replace URLs
        
    Returns:
        Tuple of (DataFrame with probability features, updated transformers dict)
    """
    from .probability_extraction import process_probability_features
    
    logger.info("="*60)
    logger.info("Processing Fine-tuned Probability Features")
    logger.info("="*60)
    
    # Step 1: Extract probability features (2D: [P(0), P(1)])
    logger.info("\n[Step 1] Extracting probability features from fine-tuned model...")
    df, transformers = process_probability_features(
        df,
        finetuned_model_path,
        is_train=is_train,
        transformers=transformers,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Step 2: Extract style features (before cleaning, if needed)
    style_features_df = pd.DataFrame(index=df.index)
    if extract_style_features:
        logger.info("\n[Step 2] Extracting text style features...")
        # Need to get raw text for style features
        fused_texts = extract_and_fuse_text_fields(df)
        style_features_df = extract_text_style_features(fused_texts)
    
    # Step 3: Extract URL features (already done in process_probability_features, but we need counts)
    url_features_df = pd.DataFrame(index=df.index)
    if classify_urls:
        logger.info("\n[Step 3] Extracting URL features...")
        fused_texts = extract_and_fuse_text_fields(df)
        _, url_features_df = classify_and_replace_urls(df, fused_texts)
    
    # Step 4: Prepare probability matrix for PCA
    logger.info("\n[Step 4] Preparing probability matrix for PCA...")
    prob_features = df[['camembert_prob_class_0', 'camembert_prob_class_1']].values
    
    # Step 5: Apply PCA reduction (from 2D to n_components)
    logger.info(f"\n[Step 5] Applying PCA reduction (2D -> {n_components}D)...")
    transformer_key = 'pca_camembert_probability'
    
    if is_train:
        # Fit PCA on training data
        pca = PCA(n_components=n_components, random_state=42)
        reduced_probs = pca.fit_transform(prob_features)
        transformers[transformer_key] = pca
        
        # Log explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
        
        if explained_var < 0.85:
            logger.warning(f"  PCA explained variance ({explained_var*100:.2f}%) is below recommended 85%")
            logger.warning(f"  Note: This is expected when reducing from 2D to {n_components}D")
    else:
        # Transform using fitted PCA
        pca = transformers.get(transformer_key)
        if pca is None:
            raise ValueError("PCA transformer not found. Must fit on training data first.")
        reduced_probs = pca.transform(prob_features)
    
    logger.info(f"Reduced probability features shape: {reduced_probs.shape}")
    
    # Step 6: Add PCA-reduced probability features to DataFrame
    logger.info("\n[Step 6] Adding PCA-reduced probability features to DataFrame...")
    # Remove original 2D probability features
    df = df.drop(columns=['camembert_prob_class_0', 'camembert_prob_class_1'], errors='ignore')
    
    # Add PCA-reduced features
    for i in range(n_components):
        col_name = f'camembert_prob_{i}'
        df[col_name] = reduced_probs[:, i]
    
    # Step 7: Add style features
    if extract_style_features and not style_features_df.empty:
        logger.info("\n[Step 7] Adding style features...")
        for col in style_features_df.columns:
            df[col] = style_features_df[col]
    
    # Step 8: Add URL features
    if classify_urls and not url_features_df.empty:
        logger.info("\n[Step 8] Adding URL features...")
        for col in url_features_df.columns:
            df[col] = url_features_df[col]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuned Probability Feature Processing Complete!")
    logger.info(f"  - Probability features (PCA): {n_components} (camembert_prob_0 to camembert_prob_{n_components-1})")
    if extract_style_features:
        logger.info(f"  - Style features: {len(style_features_df.columns)}")
    if classify_urls:
        logger.info(f"  - URL features: {len(url_features_df.columns)}")
    logger.info(f"{'='*60}\n")
    
    return df, transformers


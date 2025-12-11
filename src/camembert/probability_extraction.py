"""
Probability Feature Extraction Module


Extracts probability features from fine-tuned CamemBERT model.
"""

# 
from .._imports import (
    np, pd, Path, Dict, Tuple, Optional, logging,
    TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, torch,
    CamembertTokenizer, CamembertForSequenceClassification
)

logger = logging.getLogger(__name__)

from ..preprocessing.camembert_preprocessing import (
    extract_and_fuse_text_fields,
    classify_and_replace_urls,
    normalize_text_for_camembert
)


def load_finetuned_model(model_path: Path) -> Tuple:
    """
    Load fine-tuned CamemBERT model.
     CamemBERT 
    
    Args:
        model_path: Path to fine-tuned model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for probability extraction")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
    
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    tokenizer = CamembertTokenizer.from_pretrained(str(model_path))
    model = CamembertForSequenceClassification.from_pretrained(str(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    logger.info(f"Model loaded on device: {device}")
    
    return model, tokenizer


def extract_probability_features(
    texts: pd.Series,
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256
) -> np.ndarray:
    """
    Extract probability features from fine-tuned model.
    
    
    Args:
        texts: Series of text strings
        model: Fine-tuned CamemBERT model
        tokenizer: CamembertTokenizer
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        numpy array of shape (n_samples, 2) containing softmax probabilities [P(0), P(1)]
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for probability extraction")
    
    device = next(model.parameters()).device
    probabilities = []
    
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
            
            # Get probabilities (no gradient computation)
            with torch.no_grad():
                outputs = model(**encoded)
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                batch_probs = probs.cpu().numpy()
            
            probabilities.append(batch_probs)
            
            # Progress logging
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"  Processed {min(i + len(batch_texts), len(texts))} / {len(texts)} texts...")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"  GPU out of memory at batch {i // batch_size + 1}, reducing batch size...")
                if batch_size > 1:
                    smaller_batch = batch_size // 2
                    logger.info(f"  Retrying with batch size {smaller_batch}")
                    batch_probs = extract_probability_features(
                        texts.iloc[i:], model, tokenizer, smaller_batch, max_length
                    )
                    probabilities.append(batch_probs)
                else:
                    raise
            else:
                raise
    
    # Concatenate all batches
    if probabilities:
        all_probs = np.vstack(probabilities)
        logger.info(f"Extracted probability features shape: {all_probs.shape}")
        return all_probs
    else:
        raise ValueError("No probabilities extracted")


def process_probability_features(
    df: pd.DataFrame,
    model_path: Path,
    is_train: bool = True,
    transformers: Optional[Dict] = None,
    batch_size: int = 32,
    max_length: int = 256
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process texts to extract probability features from fine-tuned model.
    
    
    Args:
        df: Input DataFrame
        model_path: Path to fine-tuned model
        is_train: Whether this is training data
        transformers: Dictionary of transformers (for caching model)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (DataFrame with probability features, updated transformers dict)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers library not available. Skipping probability extraction.")
        return df, transformers or {}
    
    if transformers is None:
        transformers = {}
    
    logger.info("="*60)
    logger.info("Processing Probability Features")
    logger.info("="*60)
    
    # Load model (cache in transformers)
    transformer_key_model = 'finetuned_camembert_model'
    if transformer_key_model not in transformers:
        logger.info("Loading fine-tuned model...")
        model, tokenizer = load_finetuned_model(model_path)
        transformers[transformer_key_model] = {
            'model': model,
            'tokenizer': tokenizer
        }
    else:
        model_info = transformers[transformer_key_model]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        logger.info("Using cached fine-tuned model")
    
    # Step 1: Extract and fuse text fields
    logger.info("\n[Step 1] Extracting and fusing text fields...")
    fused_texts = extract_and_fuse_text_fields(df)
    
    # Step 2: Classify and replace URLs
    logger.info("\n[Step 2] Classifying and replacing URLs...")
    fused_texts, _ = classify_and_replace_urls(df, fused_texts)
    
    # Step 3: Normalize text
    logger.info("\n[Step 3] Normalizing text...")
    normalized_texts = normalize_text_for_camembert(fused_texts)
    
    # Step 4: Extract probabilities
    logger.info("\n[Step 4] Extracting probability features...")
    probabilities = extract_probability_features(
        normalized_texts, model, tokenizer, batch_size, max_length
    )
    
    # Step 5: Add probability features to DataFrame
    logger.info("\n[Step 5] Adding probability features to DataFrame...")
    df['camembert_prob_class_0'] = probabilities[:, 0]
    df['camembert_prob_class_1'] = probabilities[:, 1]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Probability Feature Extraction Complete!")
    logger.info(f"  - Added 2 probability features: camembert_prob_class_0, camembert_prob_class_1")
    logger.info(f"  - Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    logger.info(f"  - Mean probabilities: class_0={probabilities[:, 0].mean():.4f}, class_1={probabilities[:, 1].mean():.4f}")
    logger.info(f"{'='*60}\n")
    
    return df, transformers


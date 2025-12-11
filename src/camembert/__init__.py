"""
CamemBERT Embedding Module
CamemBERT 

A comprehensive module for CamemBERT-based text embedding extraction with advanced preprocessing,
style feature extraction, URL classification, and PCA dimensionality reduction.
"""

# ============================================================================
#  PyTorch  transformers
# ============================================================================
from .._imports import (
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
    torch,
    CamembertTokenizer,
    CamembertModel,
    CamembertForSequenceClassification
)

# ============================================================================
#  PyTorch 
# ============================================================================

from ..preprocessing.camembert_preprocessing import (
    extract_and_fuse_text_fields,
    extract_text_style_features,
    classify_and_replace_urls,
    normalize_text_for_camembert
)

from .embedding import (
    load_camembert_model,
    extract_camembert_embeddings,
    apply_pca_reduction,
    process_camembert_embeddings,
    process_finetuned_probability_features
)

from .finetuning import (
    create_finetuning_model,
    prepare_finetuning_data,
    train_camembert_classifier,
    save_finetuned_model,
    detect_overfitting,
    plot_finetuning_curves
)

from .probability_extraction import (
    load_finetuned_model,
    extract_probability_features,
    process_probability_features
)

# Visualization module is optional
try:
    from .visualization import (
        plot_training_curves,
        plot_embedding_quality,
        plot_feature_importance,
        plot_confusion_matrix,
        save_training_report
    )
except ImportError:
    # Visualization module not available
    plot_training_curves = None
    plot_embedding_quality = None
    plot_feature_importance = None
    plot_confusion_matrix = None
    save_training_report = None

__all__ = [
    # Preprocessing
    'extract_and_fuse_text_fields',
    'extract_text_style_features',
    'classify_and_replace_urls',
    'normalize_text_for_camembert',
    # Embedding
    'load_camembert_model',
    'extract_camembert_embeddings',
    'apply_pca_reduction',
    'process_camembert_embeddings',
    'process_finetuned_probability_features',
    # Fine-tuning
    'create_finetuning_model',
    'prepare_finetuning_data',
    'train_camembert_classifier',
    'save_finetuned_model',
    'detect_overfitting',
    'plot_finetuning_curves',
    # Probability Extraction
    'load_finetuned_model',
    'extract_probability_features',
    'process_probability_features',
    # Visualization
    'plot_training_curves',
    'plot_embedding_quality',
    'plot_feature_importance',
    'plot_confusion_matrix',
    'save_training_report',
]


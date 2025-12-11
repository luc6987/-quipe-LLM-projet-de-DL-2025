"""
CamemBERT Pipeline Main Entry Point
CamemBERT 

Command-line interface for running the complete CamemBERT embedding pipeline.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.camembert.embedding import process_camembert_embeddings
try:
    from src.vis_stat.camembert_visualization import save_training_report
except ImportError:
    save_training_report = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CamemBERT pipeline."""
    parser = argparse.ArgumentParser(
        description="CamemBERT Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='setup.yml',
        help='Path to configuration file (default: setup.yml)'
    )
    
    parser.add_argument(
        '--train-file',
        type=str,
        help='Path to training JSONL file'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        help='Path to test JSONL file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/camembert',
        help='Output directory for embeddings and visualizations'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        semantic_config = config.get('semantic', {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    logger.info("="*80)
    logger.info("CamemBERT Embedding Pipeline")
    logger.info("="*80)
    
    # Extract configuration parameters
    model_name = semantic_config.get('camembert_model_name', 'camembert-base')
    max_length = semantic_config.get('camembert_max_length', 256)
    batch_size = semantic_config.get('camembert_batch_size', 32)
    n_components = semantic_config.get('pca_components', 32)
    extract_style_features = semantic_config.get('extract_style_features', True)
    classify_urls = semantic_config.get('classify_urls', True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  PCA components: {n_components}")
    logger.info(f"  Style features: {extract_style_features}")
    logger.info(f"  URL classification: {classify_urls}")
    
    # TODO: Add data loading and processing logic here
    # This is a placeholder for the complete pipeline
    
    logger.info("Pipeline execution complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())


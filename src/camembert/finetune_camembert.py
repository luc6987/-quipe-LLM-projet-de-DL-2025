#!/usr/bin/env python3
"""
CamemBERT Fine-tuning CLI Script
CamemBERT 微调命令行脚本

Standalone script to fine-tune CamemBERT and save parameters for later use.
独立脚本用于微调 CamemBERT 并保存参数供后续使用。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.camembert.finetuning import train_camembert_classifier
from src.xgboost.utils import load_jsonl
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def save_finetuning_params(
    params: dict,
    output_path: Path
) -> None:
    """
    Save fine-tuning parameters to JSON file.
    保存微调参数到 JSON 文件。
    
    Args:
        params: Dictionary of fine-tuning parameters
        output_path: Path to save parameters JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    params['finetuned_at'] = datetime.now().isoformat()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Fine-tuning parameters saved to {output_path}")


def load_finetuning_params(params_path: Path) -> dict:
    """
    Load fine-tuning parameters from JSON file.
    从 JSON 文件加载微调参数。
    
    Args:
        params_path: Path to parameters JSON file
        
    Returns:
        Dictionary of fine-tuning parameters or None if not found
    """
    params_path = Path(params_path)
    
    if not params_path.exists():
        return None
    
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logger.info(f"Loaded fine-tuning parameters from {params_path}")
        return params
    except Exception as e:
        logger.warning(f"Failed to load fine-tuning parameters: {e}")
        return None


def main():
    """Main entry point for fine-tuning CLI."""
    parser = argparse.ArgumentParser(
        description='Fine-tune CamemBERT for classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune with default parameters
  python src/camembert/finetune_camembert.py --train-file data/train.jsonl

  # Fine-tune with custom parameters
  python src/camembert/finetune_camembert.py \\
      --train-file data/train.jsonl \\
      --epochs 5 \\
      --learning-rate 3e-5 \\
      --batch-size 32

  # Fine-tune and save to custom location
  python src/camembert/finetune_camembert.py \\
      --train-file data/train.jsonl \\
      --output-dir output/models/camembert_custom
        """
    )
    
    # Data arguments
    parser.add_argument(
        '--train-file',
        type=str,
        required=False,
        help='Path to training JSONL file (ignored if --use-preprocessed)'
    )
    parser.add_argument(
        '--use-preprocessed',
        action='store_true',
        help='Load texts/labels from output/preprocessing/ if available'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='camembert-base',
        help='Pre-trained model name (default: camembert-base)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for fine-tuned model (default: from config or output/models/camembert_finetuned)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config or 3)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: from config or 2e-5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config or 16)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Maximum sequence length (default: from config or 256)'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=None,
        help='Validation set proportion (default: from config or 0.2)'
    )
    
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help='Early stopping patience (default: from config or 2)'
    )
    
    # Output arguments
    parser.add_argument(
        '--params-file',
        type=str,
        default=None,
        help='Path to save fine-tuning parameters JSON (default: output_dir/finetuning_params.json)'
    )
    
    parser.add_argument(
        '--visualization-dir',
        type=str,
        default=None,
        help='Directory for visualization output (default: from config or output/camembert/finetuning)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable saving visualization plots'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='setup.yml',
        help='Path to configuration file (default: setup.yml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        semantic_config = config.get('semantic', {})
    except Exception as e:
        logger.warning(f"Could not load config: {e}, using defaults")
        semantic_config = {}
    
    # Set defaults from config or command line
    output_dir = args.output_dir or semantic_config.get('finetune_model_path', 'output/models/camembert_finetuned')
    num_epochs = args.epochs or semantic_config.get('finetune_epochs', 3)
    learning_rate = args.learning_rate or semantic_config.get('finetune_learning_rate', 2e-5)
    batch_size = args.batch_size or semantic_config.get('finetune_batch_size', 16)
    max_length = args.max_length or semantic_config.get('camembert_max_length', 256)
    validation_split = args.validation_split or semantic_config.get('finetune_validation_split', 0.2)
    early_stopping_patience = args.early_stopping_patience or semantic_config.get('finetune_early_stopping_patience', 2)
    visualization_dir = args.visualization_dir or semantic_config.get('finetune_visualization_dir', 'output/camembert/finetuning')
    save_plots = not args.no_plots and semantic_config.get('finetune_save_plots', True)
    
    # Determine params file path
    if args.params_file:
        params_file = Path(args.params_file)
    else:
        params_file = Path(output_dir) / 'finetuning_params.json'
    
    logger.info("="*80)
    logger.info("CamemBERT Fine-tuning CLI")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Train file: {args.train_file or 'output/preprocessing/ (preprocessed)'}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Validation split: {validation_split}")
    logger.info(f"  Early stopping patience: {early_stopping_patience}")
    logger.info(f"  Parameters file: {params_file}")
    logger.info(f"  Visualization directory: {visualization_dir}")
    logger.info("="*80)
    
    # Load training data
    preprocessed_texts = None
    if args.use_preprocessed:
        from src.preprocessing.camembert_preprocessing import (
            load_preprocessed_camembert_texts, 
            normalize_text_for_camembert,
            LINK_URL_PATTERN
        )
        logger.info("\nLoading preprocessed CamemBERT texts from output/preprocessing/ ...")
        pre_data = load_preprocessed_camembert_texts()
        if not pre_data:
            logger.error("  Error: preprocessed CamemBERT texts not found. Run main preprocessing first.")
            return 1
        # Loaded texts are fused (includes user.description), but need URL replacement and normalization
        # Apply generic URL replacement (without classification, since we don't have original DataFrame)
        logger.info("  Applying URL replacement and normalization to match inference pipeline...")
        fused_texts = pre_data['train_texts']
        # Generic URL replacement (without classification)
        texts_with_urls = fused_texts.str.replace(LINK_URL_PATTERN, '<LINK_URL>', regex=True)
        # Normalize
        preprocessed_texts = normalize_text_for_camembert(texts_with_urls)
        train_df = pd.DataFrame({
            'label': pre_data['train_labels']
        })
        logger.info(f"  Loaded {len(train_df)} samples from preprocessed artifacts")
        logger.info(f"  Using unified preprocessing pipeline (includes user.description)")
    else:
        if not args.train_file:
            logger.error("  Error: Either provide --train-file or use --use-preprocessed.")
            return 1
        logger.info(f"\nLoading training data from {args.train_file}...")
        try:
            train_df = load_jsonl(args.train_file)
            logger.info(f"  Loaded {len(train_df)} samples")
            
            # Check for label column
            if 'label' not in train_df.columns:
                logger.error("  Error: 'label' column not found in training data")
                return 1
            
            logger.info(f"  Label distribution: {train_df['label'].value_counts().to_dict()}")
        except Exception as e:
            logger.error(f"  Failed to load training data: {e}")
            return 1
    
    # Fine-tune model
    logger.info(f"\nStarting fine-tuning...")
    try:
        model, tokenizer, training_history = train_camembert_classifier(
            train_df,
            model_name=args.model_name,
            output_dir=Path(output_dir),
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            save_plots=save_plots,
            visualization_dir=Path(visualization_dir),
            preprocessed_texts=preprocessed_texts
        )
        
        logger.info(f"\n✓ Fine-tuning completed successfully!")
        
        # Save fine-tuning parameters
        finetuning_params = {
            'model_name': args.model_name,
            'output_dir': str(output_dir),
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_length': max_length,
            'validation_split': validation_split,
            'early_stopping_patience': early_stopping_patience,
            'train_file': args.train_file,
            'num_train_samples': len(train_df),
            'final_metrics': training_history.get('overfitting_info', {}),
            'model_exists': True
        }
        
        save_finetuning_params(finetuning_params, params_file)
        
        logger.info(f"\n✓ Fine-tuning parameters saved to {params_file}")
        logger.info(f"\nYou can now use this fine-tuned model by setting in setup.yml:")
        logger.info(f"  semantic:")
        logger.info(f"    use_finetuned_probabilities: true")
        logger.info(f"    finetune_model_path: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Fine-tuning failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())


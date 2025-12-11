#!/usr/bin/env python3
"""
Main Script for Distribution Shift Detection
分布偏移检测主程序

Trains a classifier to distinguish between training and test sets
to detect if feature engineering has caused distribution shift.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.temp.load_data import load_train_test_features
from src.temp.train_classifier import train_classifier
from src.temp.analyze_importance import analyze_feature_importance
from src.utils.paths import get_output_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for distribution shift detection."""
    parser = argparse.ArgumentParser(
        description='Detect distribution shift between training and test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data loading arguments
    parser.add_argument(
        '--train-features',
        type=str,
        default=None,
        help='Path to train_features.pkl (default: output/preprocessing/train_features.pkl)'
    )
    parser.add_argument(
        '--test-features',
        type=str,
        default=None,
        help='Path to test_features.pkl (default: output/preprocessing/test_features.pkl)'
    )
    parser.add_argument(
        '--balance-samples',
        action='store_true',
        help='Balance the number of samples from train and test sets'
    )
    
    # Training arguments
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='XGBoost max_depth parameter'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='XGBoost learning_rate parameter'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='XGBoost n_estimators parameter'
    )
    
    # Output arguments
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/temp)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=50,
        help='Number of top features to display in importance analysis'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Distribution Shift Detection")
    logger.info("="*80)
    logger.info(f"Arguments: {vars(args)}")
    logger.info("="*80)
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = get_output_dir('temp')
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    logger.info("\n[Step 1] Loading data...")
    train_path = Path(args.train_features) if args.train_features else None
    test_path = Path(args.test_features) if args.test_features else None
    
    try:
        X, y = load_train_test_features(
            train_features_path=train_path,
            test_features_path=test_path,
            balance_samples=args.balance_samples,
            random_state=args.random_state
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Please ensure that train_features.pkl and test_features.pkl exist.")
        logger.error("You may need to run preprocessing first.")
        return 1
    
    # Prepare XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': args.random_state,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Train classifier
    logger.info("\n[Step 2] Training classifier...")
    model_path = output_dir / 'distribution_shift_classifier.pkl' if args.save_model else None
    
    try:
        model, metrics = train_classifier(
            X=X,
            y=y,
            n_splits=args.n_splits,
            random_state=args.random_state,
            xgb_params=xgb_params,
            save_model=args.save_model,
            model_path=model_path
        )
    except ImportError as e:
        logger.error(f"Failed to train classifier: {e}")
        logger.error("Please install xgboost: pip install xgboost")
        return 1
    
    # Analyze feature importance
    logger.info("\n[Step 3] Analyzing feature importance...")
    importance_path = output_dir / 'feature_importance.csv'
    
    try:
        importance_df = analyze_feature_importance(
            model=model,
            feature_names=list(X.columns),
            top_n=args.top_n,
            save_path=importance_path
        )
    except Exception as e:
        logger.error(f"Failed to analyze feature importance: {e}")
        return 1
    
    # Save metrics summary
    logger.info("\n[Step 4] Saving results summary...")
    summary = {
        'metrics': metrics,
        'top_features': importance_df.head(args.top_n).to_dict('records'),
        'interpretation': {}
    }
    
    # Add interpretation
    auc = metrics['full_auc']
    if auc > 0.9:
        summary['interpretation'] = {
            'severity': 'HIGH',
            'message': 'Strong distribution shift detected. Feature engineering may have introduced bias.',
            'recommendation': 'Review feature engineering process, especially for top features.'
        }
    elif auc > 0.7:
        summary['interpretation'] = {
            'severity': 'MODERATE',
            'message': 'Moderate distribution shift detected. Some features may be problematic.',
            'recommendation': 'Investigate top features and consider feature selection or normalization.'
        }
    else:
        summary['interpretation'] = {
            'severity': 'LOW',
            'message': 'Minimal distribution shift detected. Features appear consistent.',
            'recommendation': 'Distribution shift is acceptable. Continue with current features.'
        }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"AUC-ROC: {metrics['full_auc']:.4f}")
    logger.info(f"Cross-validation AUC: {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
    logger.info(f"Accuracy: {metrics['full_accuracy']:.4f}")
    logger.info(f"\nInterpretation: {summary['interpretation']['severity']} - {summary['interpretation']['message']}")
    logger.info(f"Recommendation: {summary['interpretation']['recommendation']}")
    logger.info(f"\nTop 5 features contributing to distribution shift:")
    for i, row in importance_df.head(5).iterrows():
        logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.6f}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


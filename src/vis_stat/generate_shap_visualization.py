#!/usr/bin/env python3
"""
Standalone SHAP Visualization Generator
独立的SHAP可视化生成脚本

This script loads a saved XGBoost model and generates comprehensive SHAP visualizations.
此脚本加载已保存的XGBoost模型并生成全面的SHAP可视化。

Usage:
    python generate_shap_visualization.py --model-path output/models/xgboost_model_latest.pkl
    python generate_shap_visualization.py --model-path output/models/xgboost_model_20240101_120000.pkl --data-file data/train.jsonl
"""

import os
import sys

# CRITICAL: Remove ALL pytorch_packages paths from sys.path BEFORE any other imports
# This ensures we use packages from the virtual environment, not the broken pytorch_packages
sys.path = [p for p in sys.path if 'pytorch_packages' not in p]

import argparse
import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging first (before any imports that might use logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import solution modules (using virtual environment packages)
from src.xgboost.utils import load_jsonl
from src.preprocessing.xgboost_feature_engineering import engineer_features
from src.utils.config import load_config

# Import SHAP analysis module
# SHAP should be available in virtual environment
from src.vis_stat.shap_analysis import analyze_with_shap

# Import feature interaction analysis module
try:
    from src.vis_stat.feature_interaction_analysis import analyze_feature_interactions
    INTERACTION_ANALYSIS_AVAILABLE = True
except ImportError as e:
    INTERACTION_ANALYSIS_AVAILABLE = False
    logger.warning(f"Feature interaction analysis module not available: {e}")

# Import shared utilities
from src.preprocessing import data_processing as dp
from src.utils.paths import (
    get_models_output_dir, get_shap_dir, DATA_DIR, get_data_cache_dir
)


def load_model_and_metadata(model_path: str) -> tuple:
    """
    Load XGBoost model and its metadata.
    加载XGBoost模型及其元数据。
    
    Args:
        model_path: Path to the saved model (.pkl file)
        
    Returns:
        Tuple of (model, metadata_dict)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("  ✓ Model loaded")
    
    # Try to load metadata
    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    metadata = None
    
    if metadata_path.exists():
        logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info("  ✓ Metadata loaded")
    else:
        logger.warning(f"  ⚠ Metadata file not found: {metadata_path}")
        logger.info("  Will attempt to infer feature names from data")
    
    return model, metadata


def prepare_data_for_shap(
    data_file: Optional[str] = None,
    metadata: Optional[dict] = None,
    max_samples: int = 1000,
    use_cached_features: bool = True,
    return_labels: bool = False
) -> tuple:
    """
    Prepare data for SHAP analysis.
    准备用于SHAP分析的数据。
    
    Args:
        data_file: Path to data file (JSONL format)
        metadata: Model metadata containing feature names
        max_samples: Maximum number of samples to use
        use_cached_features: Whether to use cached feature-engineered data
        return_labels: Whether to return labels along with features
        
    Returns:
        Tuple of (X_data, feature_names) or (X_data, feature_names, y_data) if return_labels=True
    """
    if data_file is None:
        # Try to use default training data
        data_file = str(DATA_DIR / 'train.jsonl')
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}. Please specify --data-file"
            )
    
    logger.info(f"Loading data from: {data_file}")
    
    # Check for cached features
    cache_dir = get_data_cache_dir()
    cache_file = cache_dir / f"features_{Path(data_file).stem}.pkl"
    
    if use_cached_features and cache_file.exists():
        logger.info(f"  Loading cached features from: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            X_data = cached_data['X']
            feature_names = cached_data['feature_names']
            logger.info(f"  ✓ Cached features loaded: {X_data.shape}")
            if return_labels and 'y' in cached_data:
                # Return cached data with labels
                y_data = cached_data['y']
                # Apply sampling if needed
                if len(X_data) > max_samples:
                    logger.info(f"  Sampling {max_samples} samples from {len(X_data)} total")
                    sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
                    X_data = X_data.iloc[sample_indices]
                    if y_data is not None:
                        y_data = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
                return X_data, feature_names, y_data
            elif not return_labels:
                # Return cached data without labels
                # Apply sampling if needed
                if len(X_data) > max_samples:
                    logger.info(f"  Sampling {max_samples} samples from {len(X_data)} total")
                    sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
                    X_data = X_data.iloc[sample_indices]
                return X_data, feature_names
            else:
                # return_labels=True but labels not in cache, need to reload
                logger.warning("  ⚠ Labels not found in cache, will reload data to extract labels")
        except Exception as e:
            logger.warning(f"  ⚠ Error loading cached features: {e}")
            logger.info("  Will recompute features...")
    
    # Load raw data
    logger.info("  Loading raw data...")
    df = load_jsonl(data_file)
    logger.info(f"  Raw data shape: {df.shape}")
    
    # Extract labels if present (for feature engineering)
    y_data = None
    if 'label' in df.columns:
        y_data = df['label'].copy()
        df = df.drop(columns=['label'])
        logger.info(f"  Labels extracted: {y_data.value_counts().to_dict()}")
    
    # Load configuration for embedding parameters
    try:
        config = load_config()
        semantic_config = config.get('semantic', {})
        use_text_embedding = semantic_config.get('use_semantic_features', True)
        text_embedding_n_components = semantic_config.get('pca_components', 32)
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
        config = {}
        use_text_embedding = True
        text_embedding_n_components = 32
    
    # Apply feature engineering
    logger.info("  Applying feature engineering...")
    X_data, transformers = engineer_features(
        df,
        is_train=True,
        kaggle_max=False,
        y_train=y_data,
        use_text_embedding=use_text_embedding,
        text_embedding_n_components=text_embedding_n_components,
        config=config,
    )
    
    feature_names = X_data.columns.tolist()
    logger.info(f"  ✓ Features engineered: {X_data.shape}")
    
    # Cache features for future use (include labels if available, cache BEFORE sampling)
    # We cache the full dataset, not the sampled subset, so different sample sizes can reuse the cache
    try:
        logger.info(f"  Caching features to: {cache_file}")
        cache_dict = {
            'X': X_data,
            'feature_names': feature_names,
            'transformers': transformers
        }
        if y_data is not None:
            cache_dict['y'] = y_data
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_dict, f)
        logger.info("  ✓ Features cached" + (" (with labels)" if y_data is not None else ""))
    except Exception as e:
        logger.warning(f"  ⚠ Error caching features: {e}")
    
    # Sample data if needed (after caching)
    sample_indices = None
    if len(X_data) > max_samples:
        logger.info(f"  Sampling {max_samples} samples from {len(X_data)} total")
        sample_indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_data = X_data.iloc[sample_indices]
        if y_data is not None:
            y_data = y_data.iloc[sample_indices] if hasattr(y_data, 'iloc') else y_data[sample_indices]
    
    if return_labels:
        return X_data, feature_names, y_data
    else:
        return X_data, feature_names


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate SHAP visualizations from a saved XGBoost model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model path
    default_model = str(get_models_output_dir() / 'xgboost_model_latest.pkl')
    parser.add_argument(
        '--model-path', type=str, default=default_model,
        help='Path to saved XGBoost model (.pkl file)'
    )
    
    # Data path
    default_data = str(DATA_DIR / 'train.jsonl')
    parser.add_argument(
        '--data-file', type=str, default=default_data,
        help='Path to data file for SHAP analysis (JSONL format)'
    )
    
    # SHAP options
    parser.add_argument(
        '--max-samples', type=int, default=1000,
        help='Maximum number of samples for SHAP computation'
    )
    parser.add_argument(
        '--max-display', type=int, default=20,
        help='Maximum number of features to display in plots'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save SHAP visualizations (default: output/shap/)'
    )
    
    # Visualization options
    parser.add_argument(
        '--no-waterfall', dest='generate_waterfall', action='store_false',
        help='Skip waterfall plot generation'
    )
    parser.add_argument(
        '--no-force', dest='generate_force', action='store_false',
        help='Skip force plot generation'
    )
    parser.add_argument(
        '--no-dependence', dest='generate_dependence', action='store_false',
        help='Skip dependence plot generation'
    )
    parser.add_argument(
        '--interaction', dest='generate_interaction', action='store_true',
        help='Generate interaction plots (slower)'
    )
    parser.add_argument(
        '--no-interaction-dependence', dest='generate_interaction_dependence', action='store_false',
        help='Skip interaction dependence plot generation'
    )
    parser.add_argument(
        '--shapley-interaction-index', dest='generate_shapley_interaction_index', action='store_true',
        help='Generate Shapley Interaction Index matrix and heatmap (requires interaction values, slower)'
    )
    parser.add_argument(
        '--shap-clustering', dest='generate_shap_clustering', action='store_true',
        help='Perform SHAP-based clustering to identify heterogeneous subgroups'
    )
    parser.add_argument(
        '--num-waterfall-samples', type=int, default=5,
        help='Number of samples to generate waterfall plots for'
    )
    parser.add_argument(
        '--num-dependence-features', type=int, default=10,
        help='Number of top features to generate dependence plots for'
    )
    parser.add_argument(
        '--num-interaction-dependence-features', type=int, default=10,
        help='Number of top features to generate interaction dependence plots for'
    )
    parser.add_argument(
        '--num-clusters', type=int, default=5,
        help='Number of clusters for SHAP clustering analysis'
    )
    
    # Feature filtering options
    parser.add_argument(
        '--min-shap-threshold', type=float, default=0.001,
        help='Minimum mean absolute SHAP value threshold (features below this will be filtered out, default: 0.001)'
    )
    
    # Error analysis options
    parser.add_argument(
        '--analyze-errors', dest='analyze_errors', action='store_true',
        help='Analyze misclassified samples and generate error analysis JSON and force plots'
    )
    parser.add_argument(
        '--max-error-force-plots', type=int, default=10,
        help='Maximum number of error samples to generate force plots for (default: 10)'
    )
    
    # Cache options
    parser.add_argument(
        '--no-cache', dest='use_cached_features', action='store_false',
        help='Do not use cached feature-engineered data'
    )
    
    # Feature interaction analysis options
    parser.add_argument(
        '--analyze-interactions', dest='analyze_interactions', action='store_true',
        help='Analyze feature interactions using H-statistic and generate interaction plots'
    )
    parser.add_argument(
        '--top-n-features', type=int, default=10,
        help='Number of top features to analyze for interactions (default: 10)'
    )
    parser.add_argument(
        '--top-k-pairs', type=int, default=5,
        help='Number of top interaction pairs to select (default: 5)'
    )
    parser.add_argument(
        '--h-statistic-sample-size', type=int, default=1000,
        help='Sample size for H-statistic calculation (default: 1000)'
    )
    parser.add_argument(
        '--test-interaction-auc', dest='test_interaction_auc', action='store_true',
        help='Test AUC improvement with interaction features (requires full training data)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SHAP Visualization Generator")
    logger.info("="*80)
    
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata(args.model_path)
        
        # Prepare data (with labels if error analysis is enabled)
        prepare_result = prepare_data_for_shap(
            data_file=args.data_file,
            metadata=metadata,
            max_samples=args.max_samples,
            use_cached_features=args.use_cached_features,
            return_labels=args.analyze_errors
        )
        
        if args.analyze_errors:
            X_data, feature_names, y_data = prepare_result
            logger.info(f"  Labels loaded: {y_data.value_counts().to_dict() if hasattr(y_data, 'value_counts') else 'N/A'}")
        else:
            X_data, feature_names = prepare_result
            y_data = None
        
        # Align features with model expectations
        if metadata and 'feature_names' in metadata:
            metadata_feature_names = metadata['feature_names']
            logger.info(f"  Model expects {len(metadata_feature_names)} features")
            logger.info(f"  Data has {len(feature_names)} features")
            
            if len(metadata_feature_names) != len(feature_names) or set(metadata_feature_names) != set(feature_names):
                logger.info("  Aligning features with model expectations...")
                
                # Create a new DataFrame with model-expected features
                X_aligned = pd.DataFrame(index=X_data.index)
                
                for feature_name in metadata_feature_names:
                    if feature_name in X_data.columns:
                        # Feature exists in data, copy it
                        X_aligned[feature_name] = X_data[feature_name]
                    else:
                        # Feature missing in data, fill with 0 (or NaN for numeric)
                        logger.warning(f"    Missing feature: {feature_name}, filling with 0")
                        X_aligned[feature_name] = 0
                
                # Remove features that are not in model expectations
                extra_features = set(feature_names) - set(metadata_feature_names)
                if extra_features:
                    logger.warning(f"    Removing {len(extra_features)} extra features not in model: {list(extra_features)[:5]}...")
                
                X_data = X_aligned
                feature_names = metadata_feature_names
                logger.info(f"  ✓ Features aligned: {X_data.shape}")
            else:
                # Features match, just ensure order
                X_data = X_data[metadata_feature_names]
                feature_names = metadata_feature_names
                logger.info("  ✓ Features match model expectations")
        
        # Set output directory
        if args.output_dir is None:
            output_dir = str(get_shap_dir())
        else:
            output_dir = args.output_dir
        
        logger.info(f"\nOutput directory: {output_dir}")
        
        # Generate predictions for error analysis if needed
        y_true = None
        y_pred = None
        if args.analyze_errors:
            if y_data is None:
                logger.warning("  ⚠ Error analysis requested but labels not available in data file")
                logger.info("  Skipping error analysis. Make sure data file contains 'label' column.")
            else:
                logger.info("  Generating predictions for error analysis...")
                y_pred = model.predict(X_data.values)
                y_true = y_data.values if hasattr(y_data, 'values') else np.array(y_data)
                logger.info(f"  ✓ Predictions generated: {len(y_pred)} samples")
        
        # Generate SHAP visualizations
        logger.info("\n" + "="*80)
        logger.info("Generating SHAP Visualizations")
        logger.info("="*80)
        
        shap_result = analyze_with_shap(
            model=model,
            X_val=X_data,
            feature_names=feature_names,
            max_samples=args.max_samples,
            max_display=args.max_display,
            output_dir=output_dir,
            plot=True,
            generate_waterfall=args.generate_waterfall,
            generate_force=args.generate_force,
            generate_dependence=args.generate_dependence,
            generate_interaction=args.generate_interaction,
            generate_interaction_dependence=args.generate_interaction_dependence,
            generate_shapley_interaction_index=args.generate_shapley_interaction_index,
            generate_shap_clustering=args.generate_shap_clustering,
            num_waterfall_samples=args.num_waterfall_samples,
            num_dependence_features=args.num_dependence_features,
            num_interaction_dependence_features=args.num_interaction_dependence_features,
            num_clusters=args.num_clusters,
            min_shap_threshold=args.min_shap_threshold,
            # Error analysis parameters
            y_true=y_true,
            y_pred=y_pred,
            analyze_errors=args.analyze_errors,
            max_error_force_plots=args.max_error_force_plots
        )
        
        logger.info("\n" + "="*80)
        logger.info("SHAP Visualization Generation Complete!")
        logger.info("="*80)
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Feature importance CSV: {os.path.join(output_dir, 'feature_importance.csv')}")
        if args.analyze_errors and 'error_analysis' in shap_result and shap_result['error_analysis'] is not None:
            error_output_dir = os.path.join(output_dir, 'errors')
            logger.info(f"  Error analysis JSON: {os.path.join(error_output_dir, 'misclassified_samples.json')}")
            logger.info(f"  Error force plots: {os.path.join(error_output_dir, 'force_sample_*.png')}")
        logger.info("="*80)
        
        # Feature Interaction Analysis with H-statistic
        if args.analyze_interactions:
            if not INTERACTION_ANALYSIS_AVAILABLE:
                logger.warning("  ⚠ Feature interaction analysis not available, skipping...")
            else:
                logger.info("\n" + "="*80)
                logger.info("Feature Interaction Analysis with H-statistic")
                logger.info("="*80)
                
                try:
                    # Prepare data for interaction analysis
                    # Use the filtered SHAP data (after feature filtering) for consistency
                    shap_vals = shap_result.get('shap_values')
                    X_shap_aligned = shap_result.get('X_val_array')  # This is the aligned data for SHAP plots
                    
                    # Get the filtered feature names from SHAP result
                    # SHAP analysis may have filtered features, so we need to use the filtered list
                    if 'feature_names' in shap_result:
                        feature_names_filtered = shap_result['feature_names']
                    else:
                        # Fallback: infer from SHAP values shape
                        if shap_vals is not None:
                            # If SHAP filtered features, feature_names might be longer
                            # Use the actual number of features in SHAP values
                            feature_names_filtered = feature_names[:shap_vals.shape[1]]
                        else:
                            feature_names_filtered = feature_names
                    
                    # Use the same data that was used for SHAP (after filtering)
                    if shap_vals is not None:
                        # Get the sample indices used for SHAP
                        sample_indices = shap_result.get('sample_indices')
                        if sample_indices is not None:
                            # Use the same samples
                            X_for_interaction = X_data.iloc[sample_indices].copy()
                        else:
                            X_for_interaction = X_data.copy()
                        
                        # CRITICAL: X_for_interaction must keep ALL features (77) for partial_dependence
                        # Model expects all 77 features, so DO NOT filter X_for_interaction
                        # Only identify which features are available and non-constant for H-statistic calculation
                        
                        # Find available features (that exist in both)
                        available_features = [f for f in feature_names_filtered if f in X_for_interaction.columns]
                        if len(available_features) < len(feature_names_filtered):
                            logger.warning(
                                f"  Some features from SHAP analysis not found in X_data: "
                                f"{set(feature_names_filtered) - set(available_features)}"
                            )
                        
                        # Identify non-constant features (for selecting which pairs to calculate)
                        # But DO NOT filter X_for_interaction - it must keep all 77 features
                        non_constant_features = []
                        for feat in available_features:
                            if feat in X_for_interaction.columns:
                                feat_values = X_for_interaction[feat]
                                if feat_values.nunique() > 1:
                                    non_constant_features.append(feat)
                                else:
                                    logger.debug(
                                        f"  Feature {feat} is constant "
                                        f"(all values: {feat_values.iloc[0] if len(feat_values) > 0 else 'N/A'})"
                                    )
                        
                        if len(non_constant_features) < len(available_features):
                            logger.info(
                                f"  Found {len(non_constant_features)} features with variation "
                                f"out of {len(available_features)} available features"
                            )
                        
                        # Use non-constant features for H-statistic calculation
                        # X_for_interaction keeps ALL 77 features for model compatibility
                        feature_names_aligned = non_constant_features if non_constant_features else available_features
                        
                        # Verify X_for_interaction still has all features
                        logger.debug(
                            f"  X_for_interaction shape: {X_for_interaction.shape}, "
                            f"features for H-statistic: {len(feature_names_aligned)}"
                        )
                    else:
                        # If no SHAP values, use full data
                        X_for_interaction = X_data.copy()
                        feature_names_aligned = feature_names
                    
                    # Get feature importance path
                    feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
                    
                    # Create output directory for interactions
                    interaction_output_dir = os.path.join(output_dir, 'interactions')
                    
                    # Prepare parameters for AUC testing if requested
                    test_auc = args.test_interaction_auc
                    model_class = None
                    model_params = None
                    X_train_interaction = None
                    y_train_interaction = None
                    X_val_interaction = None
                    y_val_interaction = None
                    
                    if test_auc:
                        # Try to load full training data for AUC testing
                        logger.info("  Loading full training data for AUC testing...")
                        try:
                            # Load full training data
                            full_data_result = prepare_data_for_shap(
                                data_file=args.data_file,
                                metadata=metadata,
                                max_samples=None,  # Use all samples
                                use_cached_features=args.use_cached_features,
                                return_labels=True
                            )
                            
                            if len(full_data_result) == 3:
                                X_full, feature_names_full, y_full = full_data_result
                                
                                # Align features if needed
                                if metadata and 'feature_names' in metadata:
                                    metadata_feature_names = metadata['feature_names']
                                    if len(metadata_feature_names) == len(feature_names_full) and set(metadata_feature_names) == set(feature_names_full):
                                        X_full = X_full[metadata_feature_names]
                                        feature_names_full = metadata_feature_names
                                
                                # Split into train and validation
                                from sklearn.model_selection import train_test_split
                                X_train_interaction, X_val_interaction, y_train_interaction, y_val_interaction = train_test_split(
                                    X_full, y_full, test_size=0.2, random_state=42
                                )
                                
                                # Get model class and params from metadata
                                if metadata and 'model_params' in metadata:
                                    model_params = metadata['model_params']
                                else:
                                    # Use default XGBoost params
                                    import xgboost as xgb
                                    model_params = {
                                        'n_estimators': 100,
                                        'max_depth': 6,
                                        'learning_rate': 0.1,
                                        'objective': 'binary:logistic',
                                        'random_state': 42
                                    }
                                
                                import xgboost as xgb
                                model_class = xgb.XGBClassifier
                                
                                logger.info(f"  ✓ Prepared training data: {len(X_train_interaction)} train, {len(X_val_interaction)} val")
                            else:
                                logger.warning("  ⚠ Could not load full training data, skipping AUC test")
                                test_auc = False
                        except Exception as e:
                            logger.warning(f"  ⚠ Error preparing data for AUC test: {e}")
                            logger.info("  Continuing without AUC test...")
                            test_auc = False
                    
                    # Verify that X_for_interaction has the required features
                    # Load top features to check
                    if os.path.exists(feature_importance_path):
                        top_features_df = pd.read_csv(feature_importance_path)
                        top_features_list = top_features_df.head(args.top_n_features)['feature'].tolist()
                        
                        # Check which top features are available in X_for_interaction
                        missing_features = [f for f in top_features_list if f not in X_for_interaction.columns]
                        if missing_features:
                            logger.warning(
                                f"  Some top features not found in X_for_interaction: {missing_features[:5]}..."
                            )
                            logger.warning(
                                f"  This may be due to feature filtering. "
                                f"Available features: {len(X_for_interaction.columns)}, "
                                f"SHAP features: {len(feature_names_aligned)}"
                            )
                    
                    # Run interaction analysis
                    interaction_result = analyze_feature_interactions(
                        model=model,
                        X=X_for_interaction,  # Full 77-dim data for H-statistic calculation
                        feature_importance_path=feature_importance_path,
                        output_dir=interaction_output_dir,
                        top_n_features=args.top_n_features,
                        top_k_pairs=args.top_k_pairs,
                        grid_resolution=20,
                        sample_size=args.h_statistic_sample_size,
                        generate_plots=True,
                        test_auc=test_auc,
                        model_class=model_class,
                        model_params=model_params,
                        X_train=X_train_interaction,
                        y_train=y_train_interaction,
                        X_val=X_val_interaction,
                        y_val=y_val_interaction,
                        shap_values=shap_vals,
                        feature_names=feature_names_aligned,
                        X_shap=X_shap_aligned  # Aligned data for SHAP plots (matches shap_values)
                    )
                    
                    logger.info("\n" + "="*80)
                    logger.info("Feature Interaction Analysis Complete!")
                    logger.info("="*80)
                    logger.info(f"  Output directory: {interaction_output_dir}")
                    logger.info(f"  H-statistic matrix: {os.path.join(interaction_output_dir, 'h_statistic_matrix.csv')}")
                    logger.info(f"  Top interaction pairs: {os.path.join(interaction_output_dir, 'top_interaction_pairs.csv')}")
                    if interaction_result.get('auc_results'):
                        logger.info(f"  AUC report: {os.path.join(interaction_output_dir, 'interaction_features_auc_report.json')}")
                    logger.info("="*80)
                    
                except Exception as e:
                    logger.error(f"  ✗ Error in feature interaction analysis: {e}")
                    import traceback
                    logger.debug(f"  Full traceback: {traceback.format_exc()}")
                    logger.info("  Continuing without interaction analysis...")
        
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()


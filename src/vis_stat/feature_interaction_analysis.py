"""
Feature Interaction Analysis with H-statistic
基于 H-statistic 的特征交互分析模块

This module implements:
1. H-statistic calculation for feature pairs
2. Selection of top interaction pairs
3. SHAP Dependence Plots with interaction coloring
4. 2D Partial Dependence Plots
5. Interaction feature engineering and AUC testing
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations
import warnings

# Remove pytorch_packages from sys.path
sys.path = [p for p in sys.path if 'pytorch_packages' not in p]

try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available for H-statistic calculation")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_top_features(
    feature_importance_path: str,
    top_n: int = 10,
    available_features: Optional[List[str]] = None
) -> List[str]:
    """
    Load top N features from feature importance CSV.
    从特征重要性CSV文件加载Top N特征。
    
    Args:
        feature_importance_path: Path to feature_importance.csv
        top_n: Number of top features to select
        available_features: Optional list of available features to filter by
        
    Returns:
        List of top feature names (only those available in the data)
    """
    if not os.path.exists(feature_importance_path):
        raise FileNotFoundError(
            f"Feature importance file not found: {feature_importance_path}\n"
            "Please run SHAP analysis first to generate feature_importance.csv"
        )
    
    df = pd.read_csv(feature_importance_path)
    
    if 'feature' not in df.columns or 'mean_abs_shap' not in df.columns:
        raise ValueError(
            f"Invalid feature importance file format. "
            f"Expected columns: 'feature', 'mean_abs_shap'. "
            f"Found: {df.columns.tolist()}"
        )
    
    # Sort by importance and get top N
    top_features_all = df.nlargest(top_n, 'mean_abs_shap')['feature'].tolist()
    
    # Filter to only include features that are available in the data
    if available_features is not None:
        top_features = [f for f in top_features_all if f in available_features]
        if len(top_features) < len(top_features_all):
            missing = set(top_features_all) - set(top_features)
            logger.warning(
                f"  {len(missing)} top features not available in data: {list(missing)[:5]}..."
            )
            logger.info(
                f"  Using {len(top_features)} available features out of {len(top_features_all)} top features"
            )
    else:
        top_features = top_features_all
    
    logger.info(f"Loaded top {len(top_features)} features from {feature_importance_path}")
    logger.info(f"Top features: {top_features[:5]}...")
    
    return top_features


def calculate_h_statistic(
    model: Any,
    X: pd.DataFrame,
    feature_i: str,
    feature_j: str,
    grid_resolution: int = 20,
    sample_size: Optional[int] = None
) -> float:
    """
    Calculate H-statistic for a pair of features.
    计算一对特征的 H-statistic。
    
    H-statistic measures the strength of interaction between two features.
    H² = Var[f_ij - f_i - f_j] / Var[f_ij]
    
    where:
    - f_i: partial dependence of feature i
    - f_j: partial dependence of feature j
    - f_ij: joint partial dependence of features i and j
    
    Args:
        model: Trained model (must have predict_proba or predict method)
        X: Feature DataFrame
        feature_i: Name of first feature
        feature_j: Name of second feature
        grid_resolution: Resolution for partial dependence grid
        sample_size: Number of samples to use (None = use all)
        
    Returns:
        H-statistic value (0-1, higher means stronger interaction)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for H-statistic calculation")
    
    # Sample data if needed
    if sample_size and len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    # Get feature indices
    # Note: X might have fewer features than model expects
    # We need to find the feature indices in the model's feature space
    if feature_i not in X.columns:
        raise ValueError(f"Feature {feature_i} not found in DataFrame")
    if feature_j not in X.columns:
        raise ValueError(f"Feature {feature_j} not found in DataFrame")
    
    # Get indices in X
    feature_i_idx_in_X = X.columns.get_loc(feature_i)
    feature_j_idx_in_X = X.columns.get_loc(feature_j)
    
    # For partial_dependence, we need to use the feature names, not indices
    # because the model expects features by name or by position in its training data
    # If X has fewer features than model expects, we need to handle this differently
    
    # Try to get feature names from model if available
    try:
        # XGBoost models have get_booster().feature_names
        if hasattr(model, 'get_booster'):
            model_feature_names = model.get_booster().feature_names
            if model_feature_names:
                # Find indices in model's feature space
                if feature_i in model_feature_names:
                    feature_i_idx = model_feature_names.index(feature_i)
                else:
                    # Fallback to position in X
                    feature_i_idx = feature_i_idx_in_X
                if feature_j in model_feature_names:
                    feature_j_idx = model_feature_names.index(feature_j)
                else:
                    # Fallback to position in X
                    feature_j_idx = feature_j_idx_in_X
            else:
                # No feature names, use indices from X
                feature_i_idx = feature_i_idx_in_X
                feature_j_idx = feature_j_idx_in_X
        else:
            # Not XGBoost or no feature names, use indices from X
            feature_i_idx = feature_i_idx_in_X
            feature_j_idx = feature_j_idx_in_X
    except Exception as e:
        logger.debug(f"  Could not get model feature names: {e}, using X indices")
        feature_i_idx = feature_i_idx_in_X
        feature_j_idx = feature_j_idx_in_X
    
    # Check if features have variation
    # Use column names to access features (more reliable than indices)
    if isinstance(X_sample, pd.DataFrame):
        feat_i_values = X_sample[feature_i]
        feat_j_values = X_sample[feature_j]
    else:
        feat_i_values = pd.Series(X_sample[:, feature_i_idx])
        feat_j_values = pd.Series(X_sample[:, feature_j_idx])
    
    if feat_i_values.nunique() <= 1:
        logger.warning(
            f"  Feature {feature_i} has no variation (all values: {feat_i_values.iloc[0] if len(feat_i_values) > 0 else 'N/A'}), "
            f"skipping H-statistic calculation"
        )
        return 0.0
    
    if feat_j_values.nunique() <= 1:
        logger.warning(
            f"  Feature {feature_j} has no variation (all values: {feat_j_values.iloc[0] if len(feat_j_values) > 0 else 'N/A'}), "
            f"skipping H-statistic calculation"
        )
        return 0.0
    
    logger.debug(
        f"  Feature {feature_i}: unique values={feat_i_values.nunique()}, "
        f"range=[{feat_i_values.min():.4f}, {feat_i_values.max():.4f}]"
    )
    logger.debug(
        f"  Feature {feature_j}: unique values={feat_j_values.nunique()}, "
        f"range=[{feat_j_values.min():.4f}, {feat_j_values.max():.4f}]"
    )
    
    try:
        # Convert to numpy array for partial_dependence
        # partial_dependence needs the full feature set that model was trained on
        X_sample_array = X_sample.values
        
        # Try using feature names first (more reliable)
        try:
            pd_i = partial_dependence(
                model, X_sample, features=[feature_i],
                grid_resolution=grid_resolution,
                kind='average'
            )
        except (ValueError, TypeError, KeyError) as e:
            # If name-based fails, try index-based
            logger.debug(f"  Name-based partial_dependence failed, trying indices...")
            try:
                pd_i = partial_dependence(
                    model, X_sample_array, features=[feature_i_idx],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
            except Exception as e2:
                logger.warning(f"  Both name and index-based partial_dependence failed: {e2}")
                raise
        
        logger.debug(f"  Computing partial dependence for {feature_j}...")
        try:
            pd_j = partial_dependence(
                model, X_sample, features=[feature_j],
                grid_resolution=grid_resolution,
                kind='average'
            )
        except (ValueError, TypeError, KeyError) as e:
            pd_j = partial_dependence(
                model, X_sample_array, features=[feature_j_idx],
                grid_resolution=grid_resolution,
                kind='average'
            )
        
        # Calculate joint partial dependence
        logger.debug(f"  Computing joint partial dependence for ({feature_i}, {feature_j})...")
        try:
            pd_ij = partial_dependence(
                model, X_sample, features=[feature_i, feature_j],
                grid_resolution=grid_resolution,
                kind='average'
            )
        except (ValueError, TypeError, KeyError) as e:
            pd_ij = partial_dependence(
                model, X_sample_array, features=[feature_i_idx, feature_j_idx],
                grid_resolution=grid_resolution,
                kind='average'
            )
        
        # Check if partial dependence values are valid
        if pd_i['average'][0] is None or pd_j['average'][0] is None or pd_ij['average'][0] is None:
            logger.warning(f"  Partial dependence returned None for ({feature_i}, {feature_j})")
            return 0.0
        
        # Extract average values
        f_i = pd_i['average'][0]  # shape: (len_i,)
        f_j = pd_j['average'][0]  # shape: (len_j,)
        f_ij = pd_ij['average'][0]  # shape: (len_i, len_j)
        
        # Get grid values
        grid_i = pd_i['grid_values'][0]  # shape: (len_i,)
        grid_j = pd_j['grid_values'][0]  # shape: (len_j,)
        
        len_i = len(grid_i)
        len_j = len(grid_j)
        
        # Expand f_i and f_j to match f_ij's flattened dimensions
        # f_ij is flattened in C-order (row-major): f_ij[0,0], f_ij[0,1], ..., f_ij[0,len_j-1], f_ij[1,0], ...
        # So we need:
        # - f_i_expanded: f_i[0] repeated len_j times, then f_i[1] repeated len_j times, etc.
        # - f_j_expanded: entire f_j array repeated len_i times
        
        f_i_expanded = np.repeat(f_i, len_j)  # shape: (len_i * len_j,)
        f_j_expanded = np.tile(f_j, len_i)   # shape: (len_i * len_j,)
        
        # Flatten f_ij to match (C-order, row-major)
        f_ij_flat = f_ij.flatten()  # shape: (len_i * len_j,)
        
        # Calculate interaction effect
        interaction = f_ij_flat - f_i_expanded - f_j_expanded
        
        # Calculate H-statistic
        # H² = Var[f_ij - f_i - f_j] / Var[f_ij]
        var_interaction = np.var(interaction, ddof=0)
        var_total = np.var(f_ij_flat, ddof=0)
        
        # Debug information (use INFO level for important diagnostics)
        logger.info(
            f"  H-statistic for ({feature_i}, {feature_j}): "
            f"var_interaction={var_interaction:.6f}, var_total={var_total:.6f}, "
            f"f_i_range=[{np.min(f_i):.4f}, {np.max(f_i):.4f}], "
            f"f_j_range=[{np.min(f_j):.4f}, {np.max(f_j):.4f}], "
            f"f_ij_range=[{np.min(f_ij_flat):.4f}, {np.max(f_ij_flat):.4f}]"
        )
        
        if var_total > 1e-10:  # Use small threshold to avoid numerical issues
            H_squared = var_interaction / var_total
            logger.info(f"  → H² = {H_squared:.6f}")
        else:
            # If variance is too small, no interaction
            logger.warning(
                f"  var_total too small ({var_total:.6e}) for ({feature_i}, {feature_j}), "
                f"setting H=0.0. This may indicate constant partial dependence."
            )
            H_squared = 0.0
        
        # H-statistic should be between 0 and 1
        # Clamp to valid range (shouldn't exceed 1, but handle numerical errors)
        H_squared = max(0.0, min(1.0, H_squared))
        
        # Log debug information
        if H_squared > 0.95:
            logger.debug(
                f"  High H-statistic ({H_squared:.4f}) for ({feature_i}, {feature_j})"
            )
        elif H_squared < 0.01 and var_total > 1e-10:
            logger.debug(
                f"  Low H-statistic ({H_squared:.4f}) for ({feature_i}, {feature_j}): "
                f"weak or no interaction"
            )
        
        return float(H_squared)
        
    except Exception as e:
        logger.warning(f"  Error calculating H-statistic for ({feature_i}, {feature_j}): {e}")
        import traceback
        logger.debug(f"  Traceback: {traceback.format_exc()}")
        return 0.0


def compute_all_pairs_h_statistic(
    model: Any,
    X: pd.DataFrame,
    top_features: List[str],
    grid_resolution: int = 20,
    sample_size: Optional[int] = 1000,
    top_k: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute H-statistic for all pairs of top features.
    计算所有Top特征对的H-statistic。
    
    Args:
        model: Trained model
        X: Feature DataFrame
        top_features: List of top feature names
        grid_resolution: Resolution for partial dependence grid
        sample_size: Number of samples to use for H-statistic calculation
        top_k: Number of top interaction pairs to return
        
    Returns:
        Tuple of:
        - h_statistic_matrix: DataFrame with H-statistic values (symmetric matrix)
        - top_pairs: DataFrame with top K interaction pairs
    """
    n_features = len(top_features)
    n_pairs = n_features * (n_features - 1) // 2
    
    logger.info(f"Computing H-statistic for {n_pairs} feature pairs...")
    logger.info(f"Using sample size: {sample_size if sample_size else 'all'}")
    
    # Initialize matrix
    h_matrix = np.zeros((n_features, n_features))
    pair_results = []
    
    # Compute H-statistic for all pairs
    for idx, (feat_i, feat_j) in enumerate(combinations(top_features, 2), 1):
        logger.info(f"  [{idx}/{n_pairs}] Computing H-statistic for ({feat_i}, {feat_j})...")
        
        h_value = calculate_h_statistic(
            model, X, feat_i, feat_j,
            grid_resolution=grid_resolution,
            sample_size=sample_size
        )
        
        i_idx = top_features.index(feat_i)
        j_idx = top_features.index(feat_j)
        
        # Store in symmetric matrix
        h_matrix[i_idx, j_idx] = h_value
        h_matrix[j_idx, i_idx] = h_value
        
        pair_results.append({
            'feature_1': feat_i,
            'feature_2': feat_j,
            'h_statistic': h_value
        })
    
    # Create DataFrame for matrix
    h_statistic_matrix = pd.DataFrame(
        h_matrix,
        index=top_features,
        columns=top_features
    )
    
    # Create DataFrame for pairs
    pairs_df = pd.DataFrame(pair_results)
    pairs_df = pairs_df.sort_values('h_statistic', ascending=False)
    
    # Select top K pairs
    top_pairs = pairs_df.head(top_k).copy()
    
    logger.info(f"\nTop {len(top_pairs)} interaction pairs:")
    for idx, row in top_pairs.iterrows():
        logger.info(f"  {row['feature_1']} × {row['feature_2']}: H = {row['h_statistic']:.4f}")
    
    return h_statistic_matrix, top_pairs


def generate_shap_interaction_plots(
    model: Any,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    top_pairs: pd.DataFrame,
    output_dir: str,
    X_shap: Optional[np.ndarray] = None
) -> List[str]:
    """
    Generate SHAP Dependence Plots with interaction coloring.
    生成带交互染色的SHAP依赖图。
    
    Args:
        model: Trained model
        X: Feature DataFrame
        shap_values: SHAP values array
        feature_names: List of feature names
        top_pairs: DataFrame with top interaction pairs
        output_dir: Output directory for plots
        
    Returns:
        List of generated plot file paths
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP interaction plots")
        return []
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping plots")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    logger.info(f"Generating SHAP Dependence Plots for {len(top_pairs)} interaction pairs...")
    
    # Use X_shap if provided (aligned data), otherwise convert X
    if X_shap is not None:
        X_array = X_shap
        logger.debug(f"  Using provided X_shap with shape {X_array.shape}")
    else:
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        logger.debug(f"  Using X with shape {X_array.shape}")
    
    # Verify dimensions match
    if shap_values.shape[1] != X_array.shape[1]:
        logger.warning(
            f"  Dimension mismatch: shap_values has {shap_values.shape[1]} features, "
            f"X_array has {X_array.shape[1]} features. "
            f"This may cause incorrect plots!"
        )
        # Try to align X_array to match shap_values
        if X_array.shape[1] > shap_values.shape[1]:
            # X has more features, take first N (may not be correct!)
            logger.warning(f"  Truncating X_array from {X_array.shape[1]} to {shap_values.shape[1]} features")
            X_array = X_array[:, :shap_values.shape[1]]
        else:
            # X has fewer features, pad with zeros (may not be correct!)
            logger.warning(f"  Padding X_array from {X_array.shape[1]} to {shap_values.shape[1]} features with zeros")
            padding = np.zeros((X_array.shape[0], shap_values.shape[1] - X_array.shape[1]))
            X_array = np.hstack([X_array, padding])
    
    # Verify feature_names length matches
    if len(feature_names) != shap_values.shape[1]:
        logger.warning(
            f"  Feature names length ({len(feature_names)}) doesn't match "
            f"shap_values features ({shap_values.shape[1]}). "
            f"Truncating or extending feature_names."
        )
        if len(feature_names) > shap_values.shape[1]:
            feature_names = feature_names[:shap_values.shape[1]]
        else:
            feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), shap_values.shape[1])]
    
    for idx, row in top_pairs.iterrows():
        feat_1 = row['feature_1']
        feat_2 = row['feature_2']
        
        if feat_1 not in feature_names or feat_2 not in feature_names:
            logger.warning(f"  Skipping pair ({feat_1}, {feat_2}): features not in feature_names")
            continue
        
        feat_1_idx = feature_names.index(feat_1)
        feat_2_idx = feature_names.index(feat_2)
        
        try:
            # Plot 1: feat_1 as primary, feat_2 as interaction
            logger.info(f"  Generating plot: {feat_1} (colored by {feat_2})...")
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                feat_1_idx,
                shap_values,
                X_array,
                feature_names=feature_names,
                interaction_index=feat_2_idx,
                show=False,
                alpha=0.6
            )
            plt.title(
                f'SHAP Dependence: {feat_1}\n(Interaction with {feat_2}, H={row["h_statistic"]:.4f})',
                fontsize=12, pad=10
            )
            
            safe_feat_1 = str(feat_1).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
            safe_feat_2 = str(feat_2).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
            
            plot_path_1 = os.path.join(
                output_dir,
                f'shap_dependence_{safe_feat_1}_x_{safe_feat_2}.png'
            )
            plt.savefig(plot_path_1, bbox_inches='tight', dpi=150)
            plt.close()
            plot_paths.append(plot_path_1)
            
            # Plot 2: feat_2 as primary, feat_1 as interaction
            logger.info(f"  Generating plot: {feat_2} (colored by {feat_1})...")
            plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                feat_2_idx,
                shap_values,
                X_array,
                feature_names=feature_names,
                interaction_index=feat_1_idx,
                show=False,
                alpha=0.6
            )
            plt.title(
                f'SHAP Dependence: {feat_2}\n(Interaction with {feat_1}, H={row["h_statistic"]:.4f})',
                fontsize=12, pad=10
            )
            
            plot_path_2 = os.path.join(
                output_dir,
                f'shap_dependence_{safe_feat_2}_x_{safe_feat_1}.png'
            )
            plt.savefig(plot_path_2, bbox_inches='tight', dpi=150)
            plt.close()
            plot_paths.append(plot_path_2)
            
        except Exception as e:
            logger.warning(f"  Error generating SHAP plot for ({feat_1}, {feat_2}): {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
    
    logger.info(f"  ✓ Generated {len(plot_paths)} SHAP interaction plots")
    return plot_paths


def generate_2d_pdp(
    model: Any,
    X: pd.DataFrame,
    top_pairs: pd.DataFrame,
    output_dir: str,
    grid_resolution: int = 20
) -> List[str]:
    """
    Generate 2D Partial Dependence Plots for interaction pairs.
    为交互对生成2D偏依赖图。
    
    Args:
        model: Trained model
        X: Feature DataFrame
        top_pairs: DataFrame with top interaction pairs
        output_dir: Output directory for plots
        grid_resolution: Resolution for partial dependence grid
        
    Returns:
        List of generated plot file paths
    """
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        logger.warning("sklearn or matplotlib not available, skipping 2D PDP plots")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    logger.info(f"Generating 2D Partial Dependence Plots for {len(top_pairs)} pairs...")
    
    # Sample data if too large
    sample_size = min(1000, len(X))
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    for idx, row in top_pairs.iterrows():
        feat_1 = row['feature_1']
        feat_2 = row['feature_2']
        
        if feat_1 not in X.columns or feat_2 not in X.columns:
            logger.warning(f"  Skipping pair ({feat_1}, {feat_2}): features not in DataFrame")
            continue
        
        try:
            logger.info(f"  Generating 2D PDP for ({feat_1}, {feat_2})...")
            
            # Create 2D PDP
            display = PartialDependenceDisplay.from_estimator(
                model,
                X_sample,
                features=[feat_1, feat_2],
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            # Customize plot
            fig = display.figure_
            fig.suptitle(
                f'2D Partial Dependence: {feat_1} × {feat_2}\n(H-statistic = {row["h_statistic"]:.4f})',
                fontsize=12, y=1.02
            )
            
            safe_feat_1 = str(feat_1).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
            safe_feat_2 = str(feat_2).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
            
            plot_path = os.path.join(
                output_dir,
                f'pdp_2d_{safe_feat_1}_x_{safe_feat_2}.png'
            )
            fig.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            plot_paths.append(plot_path)
            
        except Exception as e:
            logger.warning(f"  Error generating 2D PDP for ({feat_1}, {feat_2}): {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
    
    logger.info(f"  ✓ Generated {len(plot_paths)} 2D PDP plots")
    return plot_paths


def create_interaction_features(
    X: pd.DataFrame,
    top_pairs: pd.DataFrame,
    epsilon: float = 1e-8
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create interaction features (multiplication and division) for top pairs.
    为Top交互对创建交互特征（乘法和除法）。
    
    Args:
        X: Feature DataFrame
        top_pairs: DataFrame with top interaction pairs
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Tuple of:
        - X_with_interactions: DataFrame with original and interaction features
        - interaction_feature_names: List of new interaction feature names
    """
    logger.info(f"Creating interaction features for {len(top_pairs)} pairs...")
    
    X_new = X.copy()
    interaction_feature_names = []
    
    for idx, row in top_pairs.iterrows():
        feat_1 = row['feature_1']
        feat_2 = row['feature_2']
        
        if feat_1 not in X.columns or feat_2 not in X.columns:
            logger.warning(f"  Skipping pair ({feat_1}, {feat_2}): features not in DataFrame")
            continue
        
        # Multiplication feature
        mult_name = f'{feat_1}_x_{feat_2}'
        X_new[mult_name] = X[feat_1] * X[feat_2]
        interaction_feature_names.append(mult_name)
        logger.debug(f"  Created multiplication feature: {mult_name}")
        
        # Division feature (with epsilon to avoid division by zero)
        div_name = f'{feat_1}_div_{feat_2}'
        X_new[div_name] = X[feat_1] / (X[feat_2] + epsilon)
        interaction_feature_names.append(div_name)
        logger.debug(f"  Created division feature: {div_name}")
        
        # Also create reverse division
        div_reverse_name = f'{feat_2}_div_{feat_1}'
        X_new[div_reverse_name] = X[feat_2] / (X[feat_1] + epsilon)
        interaction_feature_names.append(div_reverse_name)
        logger.debug(f"  Created reverse division feature: {div_reverse_name}")
    
    logger.info(f"  ✓ Created {len(interaction_feature_names)} interaction features")
    return X_new, interaction_feature_names


def test_auc_improvement(
    model_class: Any,
    model_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_with_interactions: pd.DataFrame,
    X_val_with_interactions: pd.DataFrame,
    interaction_feature_names: List[str]
) -> Dict[str, Any]:
    """
    Test AUC improvement with interaction features.
    测试添加交互特征后的AUC提升。
    
    Args:
        model_class: Model class (e.g., xgb.XGBClassifier)
        model_params: Model parameters dictionary
        X_train: Original training features
        y_train: Training labels
        X_val: Original validation features
        y_val: Validation labels
        X_train_with_interactions: Training features with interactions
        X_val_with_interactions: Validation features with interactions
        interaction_feature_names: List of interaction feature names
        
    Returns:
        Dictionary with AUC results and improvement metrics
    """
    logger.info("Testing AUC improvement with interaction features...")
    
    # Train baseline model (original features)
    logger.info("  Training baseline model (original features)...")
    baseline_model = model_class(**model_params)
    baseline_model.fit(X_train, y_train)
    
    val_probs_baseline = baseline_model.predict_proba(X_val)[:, 1]
    auc_baseline = roc_auc_score(y_val, val_probs_baseline)
    logger.info(f"  Baseline AUC: {auc_baseline:.6f}")
    
    # Train model with interaction features
    logger.info("  Training model with interaction features...")
    interaction_model = model_class(**model_params)
    interaction_model.fit(X_train_with_interactions, y_train)
    
    val_probs_interaction = interaction_model.predict_proba(X_val_with_interactions)[:, 1]
    auc_interaction = roc_auc_score(y_val, val_probs_interaction)
    logger.info(f"  Interaction AUC: {auc_interaction:.6f}")
    
    # Calculate improvement
    auc_improvement = auc_interaction - auc_baseline
    auc_improvement_pct = (auc_improvement / auc_baseline) * 100 if auc_baseline > 0 else 0
    
    logger.info(f"  AUC Improvement: {auc_improvement:+.6f} ({auc_improvement_pct:+.2f}%)")
    
    results = {
        'baseline_auc': float(auc_baseline),
        'interaction_auc': float(auc_interaction),
        'auc_improvement': float(auc_improvement),
        'auc_improvement_pct': float(auc_improvement_pct),
        'num_interaction_features': len(interaction_feature_names),
        'interaction_feature_names': interaction_feature_names,
        'improved': auc_improvement > 0
    }
    
    return results


def analyze_feature_interactions(
    model: Any,
    X: pd.DataFrame,
    feature_importance_path: str,
    output_dir: str,
    top_n_features: int = 10,
    top_k_pairs: int = 5,
    grid_resolution: int = 20,
    sample_size: Optional[int] = 1000,
    generate_plots: bool = True,
    test_auc: bool = True,
    model_class: Optional[Any] = None,
    model_params: Optional[Dict[str, Any]] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    X_shap: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Main function to analyze feature interactions with H-statistic.
    主函数：使用H-statistic分析特征交互。
    
    Args:
        model: Trained model
        X: Feature DataFrame (for H-statistic calculation)
        feature_importance_path: Path to feature_importance.csv
        output_dir: Output directory for results
        top_n_features: Number of top features to analyze
        top_k_pairs: Number of top interaction pairs to select
        grid_resolution: Resolution for partial dependence grid
        sample_size: Sample size for H-statistic calculation
        generate_plots: Whether to generate visualization plots
        test_auc: Whether to test AUC improvement
        model_class: Model class for AUC testing (required if test_auc=True)
        model_params: Model parameters for AUC testing (required if test_auc=True)
        X_train: Training features for AUC testing (required if test_auc=True)
        y_train: Training labels for AUC testing (required if test_auc=True)
        X_val: Validation features for AUC testing (required if test_auc=True)
        y_val: Validation labels for AUC testing (required if test_auc=True)
        shap_values: SHAP values array (optional, for SHAP plots)
        feature_names: Feature names list (optional, for SHAP plots)
        X_shap: Aligned feature array for SHAP plots (optional, should match shap_values dimensions)
        
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Feature Interaction Analysis with H-statistic")
    logger.info("="*80)
    
    # 1. Load top features
    logger.info("\n[Step 1] Loading top features...")
    # Get available features from X
    available_features = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    top_features = load_top_features(
        feature_importance_path, 
        top_n=top_n_features,
        available_features=available_features
    )
    
    # 2. Compute H-statistic for all pairs
    logger.info("\n[Step 2] Computing H-statistic for all feature pairs...")
    h_matrix, top_pairs = compute_all_pairs_h_statistic(
        model, X, top_features,
        grid_resolution=grid_resolution,
        sample_size=sample_size,
        top_k=top_k_pairs
    )
    
    # Save H-statistic matrix and top pairs
    h_matrix_path = os.path.join(output_dir, 'h_statistic_matrix.csv')
    h_matrix.to_csv(h_matrix_path)
    logger.info(f"  ✓ H-statistic matrix saved: {h_matrix_path}")
    
    top_pairs_path = os.path.join(output_dir, 'top_interaction_pairs.csv')
    top_pairs.to_csv(top_pairs_path, index=False)
    logger.info(f"  ✓ Top interaction pairs saved: {top_pairs_path}")
    
    # 3. Generate visualizations
    shap_plot_paths = []
    pdp_plot_paths = []
    
    if generate_plots:
        # SHAP Dependence Plots
        if shap_values is not None and feature_names is not None:
            logger.info("\n[Step 3a] Generating SHAP Dependence Plots...")
            shap_plot_paths = generate_shap_interaction_plots(
                model, X, shap_values, feature_names, top_pairs, output_dir, X_shap=X_shap
            )
        else:
            logger.warning("  Skipping SHAP plots: shap_values or feature_names not provided")
        
        # 2D PDP Plots
        logger.info("\n[Step 3b] Generating 2D Partial Dependence Plots...")
        pdp_plot_paths = generate_2d_pdp(
            model, X, top_pairs, output_dir, grid_resolution=grid_resolution
        )
    
    # 4. Create interaction features and test AUC
    auc_results = None
    if test_auc:
        if not all([model_class, model_params, X_train is not None, y_train is not None,
                   X_val is not None, y_val is not None]):
            logger.warning("  Skipping AUC test: required parameters not provided")
        else:
            logger.info("\n[Step 4] Creating interaction features and testing AUC...")
            
            # Create interaction features
            X_train_with_interactions, interaction_feature_names = create_interaction_features(
                X_train, top_pairs
            )
            X_val_with_interactions, _ = create_interaction_features(X_val, top_pairs)
            
            # Test AUC improvement
            auc_results = test_auc_improvement(
                model_class, model_params,
                X_train, y_train, X_val, y_val,
                X_train_with_interactions, X_val_with_interactions,
                interaction_feature_names
            )
            
            # Save AUC results
            auc_report_path = os.path.join(output_dir, 'interaction_features_auc_report.json')
            with open(auc_report_path, 'w', encoding='utf-8') as f:
                json.dump(auc_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  ✓ AUC report saved: {auc_report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Feature Interaction Analysis Complete!")
    logger.info("="*80)
    
    results = {
        'top_features': top_features,
        'h_statistic_matrix': h_matrix,
        'top_interaction_pairs': top_pairs,
        'shap_plot_paths': shap_plot_paths,
        'pdp_plot_paths': pdp_plot_paths,
        'auc_results': auc_results
    }
    
    return results


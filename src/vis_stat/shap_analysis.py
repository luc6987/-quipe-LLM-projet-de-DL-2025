"""
SHAP Analysis Module
SHAP分析模块 / Module d'analyse SHAP / SHAP Analysis Module / SHAP-Analysemodul

Implements Phase 4: Feature Interpretability using SHAP (SHapley Additive exPlanations).
实现第四阶段：使用SHAP进行特征可解释性分析 / Implements Phase 4: Feature Interpretability using SHAP (SHapley Additive exPlanations)

SHAP Values:
- Based on Shapley values from cooperative game theory
- Quantifies each feature's contribution to prediction
- Provides local (per-sample) and global (aggregate) interpretability

Etymology: "Shapley" from Lloyd Shapley (1923-2016), Nobel Prize winner in Economics
for work on cooperative game theory and fair value distribution.
"""

import os
import sys

# Remove pytorch_packages from sys.path BEFORE any imports to avoid conflicts
sys.path = [p for p in sys.path if 'pytorch_packages' not in p]

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
import json

# Try importing SHAP from multiple locations
SHAP_AVAILABLE = False
shap = None

# Method 1: Try standard import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Method 2: Try importing from pytorch_packages if standard import failed
if not SHAP_AVAILABLE and os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    try:
        import importlib.util
        shap_path = '/Data/yuguang.yao/pytorch_packages/shap'
        if os.path.exists(shap_path):
            spec = importlib.util.spec_from_file_location("shap", os.path.join(shap_path, "__init__.py"))
            if spec and spec.loader:
                shap = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(shap)
                SHAP_AVAILABLE = True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not import SHAP from pytorch_packages: {e}")

# Method 3: Try importing as a package in pytorch_packages
if not SHAP_AVAILABLE:
    try:
        import importlib
        shap = importlib.import_module('shap')
        SHAP_AVAILABLE = True
    except ImportError:
        pass

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Initialize logger after imports
logger = logging.getLogger(__name__)

# Log SHAP availability status
if SHAP_AVAILABLE:
    try:
        logger.debug(f"SHAP imported successfully, version: {getattr(shap, '__version__', 'unknown')}")
    except:
        pass
else:
    logger.debug("SHAP not available - will skip SHAP analysis")


def analyze_with_shap(model: Any, X_val: pd.DataFrame, feature_names: Optional[List[str]] = None,
                      max_samples: int = 1000, max_display: int = 20,
                      output_dir: Optional[str] = None, plot: bool = True,
                      generate_waterfall: bool = True, generate_force: bool = True,
                      generate_dependence: bool = True, generate_interaction: bool = False,
                      generate_interaction_dependence: bool = True,
                      generate_shapley_interaction_index: bool = False,
                      generate_shap_clustering: bool = False,
                      num_waterfall_samples: int = 5, num_dependence_features: int = 10,
                      num_interaction_dependence_features: int = 10,
                      num_clusters: int = 5,
                      min_shap_threshold: float = 0.001,
                      y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None,
                      analyze_errors: bool = False,
                      max_error_force_plots: int = 10) -> Dict[str, Any]:
    """
    Perform comprehensive SHAP analysis for feature interpretability.
    执行全面的SHAP分析以进行特征可解释性分析。
    
    Generates:
    1. SHAP summary plot (beeswarm plot) - shows feature importance and impact
    2. Bar plot - feature importance ranking
    3. Waterfall plot - individual sample feature contributions
    4. Force plot - interactive feature impact visualization
    5. Dependence plot - feature value vs SHAP value relationships
    6. Interaction plot - feature interaction analysis (optional)
    7. Interaction Dependence Plot - detailed feature interaction visualization
    8. Shapley Interaction Index - quantitative interaction strength matrix (optional)
    9. SHAP Clustering - identify heterogeneous subgroups with different feature patterns (optional)
    
    Args:
        model: Trained XGBoost model
        X_val: Validation set features DataFrame
        feature_names: List of feature names (if None, uses DataFrame columns)
        max_samples: Maximum number of samples for SHAP calculation (for speed)
        max_display: Maximum number of features to display in plots
        output_dir: Directory to save SHAP plots and CSV files
        plot: Whether to generate visualization plots
        generate_waterfall: Whether to generate waterfall plots for individual samples
        generate_force: Whether to generate force plots (HTML format)
        generate_dependence: Whether to generate dependence plots
        generate_interaction: Whether to compute and plot interaction values (slower)
        generate_interaction_dependence: Whether to generate interaction dependence plots
        generate_shapley_interaction_index: Whether to compute and visualize Shapley Interaction Index (requires interaction values, slower)
        generate_shap_clustering: Whether to perform SHAP-based clustering to identify heterogeneous subgroups
        num_waterfall_samples: Number of samples to generate waterfall plots for
        num_clusters: Number of clusters for SHAP clustering analysis
        num_dependence_features: Number of top features to generate dependence plots for
        num_interaction_dependence_features: Number of top features to generate interaction dependence plots for
        min_shap_threshold: Minimum mean absolute SHAP value threshold (features below this will be filtered out)
        y_true: True labels array (optional, required if analyze_errors=True)
        y_pred: Predicted labels array (optional, required if analyze_errors=True)
        analyze_errors: Whether to analyze misclassified samples (default: False)
        max_error_force_plots: Maximum number of error samples to generate force plots for (default: 10)
        
    Returns:
        Dictionary containing:
        - 'shap_values': SHAP values array
        - 'feature_importance': DataFrame with feature importance rankings
        - 'explainer': SHAP TreeExplainer object
        - 'sample_indices': Indices of samples used (if sampling was applied)
        - 'base_value': Expected value from the model
        - 'error_analysis': Dictionary with misclassified samples analysis (if analyze_errors=True)
    """
    if not SHAP_AVAILABLE or shap is None:
        error_msg = "SHAP is not available. "
        if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
            error_msg += "SHAP may be installed in /Data/yuguang.yao/pytorch_packages but could not be imported. "
        error_msg += "Please ensure SHAP is properly installed: pip install shap"
        raise ImportError(error_msg)
    
    # Use unified path configuration if output_dir is not specified
    if output_dir is None:
        from src.utils.paths import get_shap_dir
        output_dir = str(get_shap_dir())
    
    logger.info("="*60)
    logger.info("SHAP Analysis: Feature Interpretability")
    logger.info("="*60)
    
    # Convert to numpy array
    if isinstance(X_val, pd.DataFrame):
        X_val_array = X_val.values
        if feature_names is None:
            feature_names = X_val.columns.tolist()
    else:
        X_val_array = X_val
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_val_array.shape[1])]
    
    # Limit samples for faster computation
    if len(X_val_array) > max_samples:
        logger.info(f"  Limiting to {max_samples} samples for faster SHAP computation")
        sample_indices = np.random.choice(len(X_val_array), max_samples, replace=False)
        X_val_array = X_val_array[sample_indices]
    else:
        sample_indices = None
    
    # Create SHAP TreeExplainer for XGBoost
    # TreeExplainer is optimized for tree-based models (XGBoost, LightGBM, etc.)
    logger.info("  Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Get base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    # Calculate SHAP values
    logger.info("  Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_val_array)
    
    # Handle binary classification: SHAP returns list for binary, single array for regression
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    logger.info(f"  ✓ SHAP values computed: shape {shap_values.shape}")
    logger.info(f"  Base value (expected value): {base_value:.4f}")
    
    # Calculate feature importance (mean absolute SHAP value)
    # This is more reliable than XGBoost's built-in importance for interpretability
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Filter features with SHAP value below threshold
    original_feature_count = len(feature_importance)
    
    if min_shap_threshold > 0:
        feature_importance_filtered = feature_importance[feature_importance['mean_abs_shap'] >= min_shap_threshold].copy()
        filtered_feature_count = len(feature_importance_filtered)
        removed_count = original_feature_count - filtered_feature_count
        
        if removed_count > 0:
            logger.info(f"\n  Filtering features: {removed_count} features with mean_abs_shap < {min_shap_threshold} removed")
            logger.info(f"  Remaining features: {filtered_feature_count} (from {original_feature_count})")
            
            # Get feature names to keep
            keep_feature_names = feature_importance_filtered['feature'].tolist()
            
            # Get column indices of features to keep
            keep_indices = [feature_names.index(fname) for fname in keep_feature_names]
            
            # Filter shap_values, X_val_array, and feature_names
            shap_values = shap_values[:, keep_indices]
            X_val_array = X_val_array[:, keep_indices]
            feature_names = keep_feature_names
            feature_importance = feature_importance_filtered.reset_index(drop=True)
        else:
            logger.info(f"\n  No features filtered (all features have mean_abs_shap >= {min_shap_threshold})")
    else:
        # min_shap_threshold=0 means no filtering
        logger.info(f"\n  No feature filtering (min_shap_threshold=0, keeping all {original_feature_count} features)")
    
    logger.info(f"\n  Top {min(max_display, len(feature_importance))} Important Features:")
    for idx, row in feature_importance.head(max_display).iterrows():
        logger.info(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Save full feature importance to CSV for Plan Optimizer
    os.makedirs(output_dir, exist_ok=True)
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_csv_path, index=False)
    logger.info(f"  ✓ Feature importance saved to: {importance_csv_path}")
    
    # Save feature importance to JSON for feature filtering
    importance_json_path = os.path.join(output_dir, 'feature_importance.json')
    shap_dict = dict(zip(feature_importance['feature'], feature_importance['mean_abs_shap']))
    with open(importance_json_path, 'w', encoding='utf-8') as f:
        json.dump(shap_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Feature importance saved to JSON: {importance_json_path}")

    # Initialize interaction-related variables (for return value)
    interaction_values_result = None
    interaction_sample_indices_result = None
    interaction_index_matrix_result = None
    interaction_pairs_result = None
    top_interaction_features_selected = None  # Store selected feature names for interaction index
    cluster_labels_result = None  # Store cluster assignments
    
    # Generate visualizations
    if plot and MATPLOTLIB_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Summary Plot (Beeswarm Plot)
            # Shows feature importance and impact direction
            logger.info("\n  Generating SHAP summary plot (beeswarm)...")
            plt.figure(figsize=(10, max(6, max_display * 0.4)))
            shap.summary_plot(
                shap_values,
                X_val_array,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            summary_plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
            plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"  ✓ Summary plot saved: {summary_plot_path}")
            
            # 2. Bar Plot (Feature Importance)
            # Bar plot shows the mean absolute SHAP value for each feature
            # 条形图显示每个特征的平均绝对SHAP值（特征重要性排序）
            logger.info("  Generating SHAP bar plot...")
            bar_plot_path = os.path.join(output_dir, 'shap_bar_plot.png')
            try:
                # Create Explanation object for better compatibility
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=np.full(len(shap_values), base_value),
                    data=X_val_array,
                    feature_names=feature_names
                )
                
                # Try new API first (shap >= 0.40) with matplotlib=True
                try:
                    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
                    shap.plots.bar(
                        explanation,
                        max_display=max_display,
                        show=False
                    )
                    plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
                    plt.close()
                    logger.info(f"  ✓ Bar plot saved: {bar_plot_path}")
                except Exception as e1:
                    logger.debug(f"  New API failed: {e1}")
                    try:
                        # Alternative: use summary_plot with plot_type='bar'
                        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
                        shap.summary_plot(
                            shap_values,
                            X_val_array,
                            feature_names=feature_names,
                            max_display=max_display,
                            plot_type='bar',
                            show=False
                        )
                        plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(f"  ✓ Bar plot saved (using summary_plot): {bar_plot_path}")
                    except Exception as e2:
                        logger.debug(f"  Summary plot bar failed: {e2}")
                        # Fallback: manual bar plot using feature importance
                        try:
                            fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))
                            top_features = feature_importance.head(max_display)
                            y_pos = np.arange(len(top_features))
                            ax.barh(y_pos, top_features['mean_abs_shap'].values, align='center')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(top_features['feature'].values)
                            ax.invert_yaxis()
                            ax.set_xlabel('Mean |SHAP value| (average impact on model output magnitude)')
                            ax.set_title('SHAP Feature Importance')
                            plt.tight_layout()
                            plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"  ✓ Bar plot saved (manual): {bar_plot_path}")
                        except Exception as e3:
                            logger.warning(f"  ⚠ All bar plot methods failed: {e3}")
                            try:
                                plt.close()
                            except:
                                pass
            except Exception as e:
                logger.warning(f"  ⚠ Error generating bar plot: {e}")
                try:
                    plt.close()
                except:
                    pass
            
            # 3. Waterfall Plot (Individual Sample Analysis)
            if generate_waterfall and num_waterfall_samples > 0:
                logger.info(f"  Generating waterfall plots for {num_waterfall_samples} samples...")
                num_samples_to_plot = min(num_waterfall_samples, len(X_val_array))
                sample_indices_for_waterfall = np.random.choice(
                    len(X_val_array), num_samples_to_plot, replace=False
                )
                
                for idx, sample_idx in enumerate(sample_indices_for_waterfall):
                    try:
                        plt.figure(figsize=(10, 8))
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[sample_idx],
                                base_values=base_value,
                                data=X_val_array[sample_idx],
                                feature_names=feature_names
                            ),
                            max_display=max_display,
                            show=False
                        )
                        
                        waterfall_path = os.path.join(
                            output_dir, f'shap_waterfall_sample_{sample_idx}.png'
                        )
                        plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(f"    ✓ Waterfall plot {idx+1}/{num_samples_to_plot} saved: {waterfall_path}")
                    except Exception as e:
                        logger.warning(f"    ⚠ Error generating waterfall plot for sample {sample_idx}: {e}")
            
            # 4. Force Plot (Individual Sample Feature Contributions)
            # Force plot shows how each feature value pushes the prediction away from the base value
            # 力图显示每个特征值如何将预测值从基准值推离（红色=增加预测值，蓝色=减少预测值）
            # True force plot style: horizontal cumulative bars showing base -> features -> prediction
            if generate_force:
                logger.info("  Generating force plots...")
                try:
                    # Generate force plot for a few representative samples
                    num_force_samples = min(10, len(X_val_array))
                    force_indices = np.random.choice(len(X_val_array), num_force_samples, replace=False)
                    
                    for idx, sample_idx in enumerate(force_indices):
                        try:
                            # Use idx for filename to ensure files are overwritten on each run
                            # The actual sample_idx is shown in the plot annotation
                            force_png_path = os.path.join(
                                output_dir, f'shap_force_sample_{idx}.png'
                            )
                            force_html_path = os.path.join(
                                output_dir, f'shap_force_sample_{idx}.html'
                            )
                            
                            success = False
                            
                            # Method 1: Try real SHAP force plot with matplotlib=True (if available)
                            try:
                                fig = plt.figure(figsize=(12, 6))
                                shap.plots.force(
                                    base_value,
                                    shap_values[sample_idx],
                                    X_val_array[sample_idx],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False,
                                    fig=fig
                                )
                                # Add sample index annotation
                                fig.text(0.98, 0.98, f'Sample: {sample_idx}', 
                                        transform=fig.transFigure,
                                        fontsize=10, fontweight='bold',
                                        ha='right', va='top',
                                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                                plt.savefig(force_png_path, bbox_inches='tight', dpi=150)
                                plt.close()
                                logger.info(f"    ✓ Force plot {idx+1}/{num_force_samples} saved (SHAP force plot PNG): {force_png_path}")
                                success = True
                            except Exception as e1:
                                logger.debug(f"    SHAP matplotlib force plot failed: {e1}")
                                try:
                                    plt.close()
                                except:
                                    pass
                            
                            # Method 2: Manual horizontal bar plot (true force plot style)
                            # This shows cumulative effect: base value -> feature contributions -> final prediction
                            if not success:
                                try:
                                    fig, ax = plt.subplots(figsize=(14, max(6, max_display * 0.4)))
                                    sample_shap = shap_values[sample_idx]
                                    sample_data = X_val_array[sample_idx]
                                    
                                    # Sort features by absolute SHAP value for better visualization
                                    feature_df = pd.DataFrame({
                                        'feature': feature_names,
                                        'shap_value': sample_shap,
                                        'feature_value': sample_data,
                                        'abs_shap': np.abs(sample_shap)
                                    }).sort_values('abs_shap', ascending=False).head(max_display)
                                    
                                    # Create horizontal bar chart showing SHAP contributions
                                    y_pos = np.arange(len(feature_df))
                                    colors = ['red' if x > 0 else 'blue' for x in feature_df['shap_value'].values]
                                    
                                    # Draw bars showing SHAP contributions (horizontal)
                                    bars = ax.barh(y_pos, feature_df['shap_value'].values, 
                                                  align='center', color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                                    
                                    # Add feature values to labels for context
                                    labels = []
                                    for fname, val, shap_val in zip(feature_df['feature'].values, 
                                                                   feature_df['feature_value'].values,
                                                                   feature_df['shap_value'].values):
                                        # Format feature value appropriately
                                        if isinstance(val, (int, float)):
                                            if abs(val) < 0.01:
                                                val_str = f"{val:.4f}"
                                            elif abs(val) < 1:
                                                val_str = f"{val:.3f}"
                                            else:
                                                val_str = f"{val:.2f}"
                                        else:
                                            val_str = str(val)
                                        labels.append(f"{fname}\n(val={val_str}, SHAP={shap_val:.3f})")
                                    
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(labels, fontsize=8)
                                    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
                                    ax.set_xlabel('SHAP Value (contribution to model output)', fontsize=10, fontweight='bold')
                                    final_pred = base_value + sample_shap.sum()
                                    ax.set_title(f'Force Plot - Sample {sample_idx}\nBase value: {base_value:.4f} → Final prediction: {final_pred:.4f}', 
                                                fontsize=11, pad=10, fontweight='bold')
                                    
                                    # Add text annotations for base value and final prediction
                                    ax.text(base_value, len(feature_df) - 0.5, f'Base\n{base_value:.4f}', 
                                           ha='center', va='top', fontsize=8, 
                                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                                    
                                    # Add sample index annotation
                                    ax.text(0.98, 0.98, f'Sample: {sample_idx}', 
                                           transform=ax.transAxes,
                                           fontsize=10, fontweight='bold',
                                           ha='right', va='top',
                                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                                    
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    plt.savefig(force_png_path, bbox_inches='tight', dpi=150)
                                    plt.close()
                                    logger.info(f"    ✓ Force plot {idx+1}/{num_force_samples} saved (manual force plot): {force_png_path}")
                                    success = True
                                except Exception as e2:
                                    logger.debug(f"    Manual force plot failed: {e2}")
                                    try:
                                        plt.close()
                                    except:
                                        pass
                            
                            # Method 3: Try HTML force plot (interactive, as backup)
                            if not success:
                                try:
                                    shap.initjs()
                                    html_str = shap.force_plot(
                                        base_value,
                                        shap_values[sample_idx],
                                        X_val_array[sample_idx],
                                        feature_names=feature_names,
                                        matplotlib=False,
                                        show=False
                                    )
                                    # SHAP force_plot returns an HTML wrapper, extract actual HTML
                                    if hasattr(html_str, 'data'):
                                        html_content = html_str.data
                                    elif isinstance(html_str, str):
                                        html_content = html_str
                                    else:
                                        html_content = str(html_str)
                                    
                                    with open(force_html_path, 'w', encoding='utf-8') as f:
                                        f.write(html_content)
                                    logger.info(f"    ✓ Force plot {idx+1}/{num_force_samples} saved (HTML): {force_html_path}")
                                    # Also try to create a simple PNG visualization
                                    try:
                                        fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.4)))
                                        sample_shap = shap_values[sample_idx]
                                        feature_importance_sample = pd.DataFrame({
                                            'feature': feature_names,
                                            'shap_value': sample_shap,
                                            'abs_shap': np.abs(sample_shap)
                                        }).sort_values('abs_shap', ascending=False).head(max_display)
                                        
                                        colors = ['red' if x > 0 else 'blue' for x in feature_importance_sample['shap_value'].values]
                                        y_pos = np.arange(len(feature_importance_sample))
                                        bars = ax.barh(y_pos, feature_importance_sample['shap_value'].values, 
                                                      align='center', color=colors, alpha=0.7)
                                        ax.set_yticks(y_pos)
                                        ax.set_yticklabels(feature_importance_sample['feature'].values)
                                        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                                        ax.set_xlabel('SHAP value (impact on model output)')
                                        final_pred = base_value + sample_shap.sum()
                                        ax.set_title(f'Force Plot - Sample {sample_idx}\nBase: {base_value:.4f} → Final: {final_pred:.4f}')
                                        # Add sample index annotation
                                        ax.text(0.98, 0.98, f'Sample: {sample_idx}', 
                                               transform=ax.transAxes,
                                               fontsize=10, fontweight='bold',
                                               ha='right', va='top',
                                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                                        ax.invert_yaxis()
                                        plt.tight_layout()
                                        plt.savefig(force_png_path, bbox_inches='tight', dpi=150)
                                        plt.close()
                                        logger.info(f"    ✓ Force plot PNG also saved: {force_png_path}")
                                    except:
                                        pass
                                    success = True
                                except Exception as e3:
                                    logger.debug(f"    HTML method failed: {e3}")
                            
                            # Method 4: Simple bar chart as last resort
                            if not success:
                                try:
                                    fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.4)))
                                    sample_shap = shap_values[sample_idx]
                                    feature_importance_sample = pd.DataFrame({
                                        'feature': feature_names,
                                        'shap_value': sample_shap,
                                        'abs_shap': np.abs(sample_shap)
                                    }).sort_values('abs_shap', ascending=False).head(max_display)
                                    
                                    colors = ['red' if x > 0 else 'blue' for x in feature_importance_sample['shap_value'].values]
                                    y_pos = np.arange(len(feature_importance_sample))
                                    bars = ax.barh(y_pos, feature_importance_sample['shap_value'].values, 
                                                  align='center', color=colors, alpha=0.7)
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(feature_importance_sample['feature'].values)
                                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                                    ax.set_xlabel('SHAP value (impact on model output)')
                                    final_pred = base_value + sample_shap.sum()
                                    ax.set_title(f'Force Plot - Sample {sample_idx}\nBase: {base_value:.4f} → Final: {final_pred:.4f}')
                                    # Add sample index annotation
                                    ax.text(0.98, 0.98, f'Sample: {sample_idx}', 
                                           transform=ax.transAxes,
                                           fontsize=10, fontweight='bold',
                                           ha='right', va='top',
                                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                                    ax.invert_yaxis()
                                    plt.tight_layout()
                                    plt.savefig(force_png_path, bbox_inches='tight', dpi=150)
                                    plt.close()
                                    logger.info(f"    ✓ Force plot {idx+1}/{num_force_samples} saved (simple bar chart): {force_png_path}")
                                except Exception as e4:
                                    logger.warning(f"    ⚠ All force plot methods failed for sample {sample_idx}: {e4}")
                                    try:
                                        plt.close()
                                    except:
                                        pass
                            
                        except Exception as sample_error:
                            logger.warning(f"    ⚠ Error processing sample {sample_idx}: {sample_error}")
                            
                except Exception as e:
                    logger.warning(f"  ⚠ Error generating force plots: {e}")
            
            # 5. Dependence Plot (Feature Value vs SHAP Value)
            # Dependence plots show how a single feature affects the model's output
            # X-axis: feature value, Y-axis: SHAP value for that feature
            # Color: values of another feature (shows interactions)
            # 
            # How to interpret:
            # - Vertical spread: indicates interaction effects (if color varies vertically at same x-value)
            # - Trend: shows if feature has linear or non-linear relationship with output
            # - Clusters: groups of samples with similar behavior
            # 
            # 依赖图显示单个特征如何影响模型输出
            # X轴：特征值，Y轴：该特征的SHAP值
            # 颜色：另一个特征的值（显示交互作用）
            # 
            # 如何解读：
            # - 垂直分布：表示交互效应（如果在相同x值处颜色垂直变化）
            # - 趋势：显示特征与输出是线性还是非线性关系
            # - 聚类：具有相似行为的样本组
            if generate_dependence and num_dependence_features > 0:
                logger.info(f"  Generating dependence plots for top {num_dependence_features} features...")
                top_features = feature_importance.head(num_dependence_features)['feature'].tolist()
                
                for idx, feature in enumerate(top_features):
                    try:
                        if feature in feature_names:
                            feature_idx = feature_names.index(feature)
                            
                            # Find feature with strongest interaction (use second most important that's different)
                            # This helps reveal feature interactions by coloring points with another feature's value
                            interaction_feature = None
                            interaction_idx = None
                            if len(top_features) > 1:
                                for other_feature in top_features:
                                    if other_feature != feature and other_feature in feature_names:
                                        interaction_feature = other_feature
                                        interaction_idx = feature_names.index(other_feature)
                                        break
                            
                            plt.figure(figsize=(10, 7))
                            
                            # Use dependence_plot
                            # If interaction_index is specified, points are colored by that feature's value
                            # This reveals how the two features interact
                            if interaction_feature and interaction_idx is not None:
                                shap.dependence_plot(
                                    feature_idx,  # Primary feature (X-axis)
                                    shap_values,  # SHAP values for all samples
                                    X_val_array,  # Feature values for all samples
                                    feature_names=feature_names,  # Feature names
                                    interaction_index=interaction_idx,  # Feature to show interaction with (color)
                                    show=False,
                                    alpha=0.6  # Transparency for better visibility
                                )
                                plt.title(f'SHAP Dependence Plot: {feature}\n(Colored by {interaction_feature})', 
                                         fontsize=12, pad=10)
                            else:
                                shap.dependence_plot(
                                    feature_idx,
                                    shap_values,
                                    X_val_array,
                                    feature_names=feature_names,
                                    show=False,
                                    alpha=0.6
                                )
                                plt.title(f'SHAP Dependence Plot: {feature}', fontsize=12, pad=10)
                            
                            plt.xlabel(f'{feature} (feature value)', fontsize=10)
                            plt.ylabel(f'SHAP value for {feature}', fontsize=10)
                            
                            # Sanitize filename (replace special characters) - only sanitize the filename part
                            safe_feature_name = str(feature).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
                            dep_path = os.path.join(output_dir, f'shap_dependence_{safe_feature_name}.png')
                            plt.savefig(dep_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"    ✓ Dependence plot {idx+1}/{len(top_features)} saved: {dep_path}")
                    except Exception as e:
                        logger.warning(f"    ⚠ Error generating dependence plot for {feature}: {e}")
                        import traceback
                        logger.debug(f"    Traceback: {traceback.format_exc()}")
            
            # 6. Interaction Values Plot (optional, slower)
            # Store interaction values for Shapley Interaction Index computation
            if generate_interaction or generate_shapley_interaction_index:
                logger.info("  Computing interaction values (this may take a while)...")
                try:
                    # Compute interaction values for top features only (to save time)
                    # For Shapley Interaction Index, we need all feature pairs, so use more features if needed
                    if generate_shapley_interaction_index:
                        # Use more features for interaction index computation
                        top_interaction_features = feature_importance.head(min(20, len(feature_names)))['feature'].tolist()
                    else:
                        top_interaction_features = feature_importance.head(min(10, max_display))['feature'].tolist()
                    
                    interaction_indices = [feature_names.index(f) for f in top_interaction_features if f in feature_names]
                    
                    if len(interaction_indices) > 0:
                        # Sample data for interaction computation (can be slow)
                        interaction_sample_size = min(500, len(X_val_array))
                        interaction_sample_idx = np.random.choice(
                            len(X_val_array), interaction_sample_size, replace=False
                        )
                        
                        # Compute interaction values for all features
                        # SHAP interaction values are computed for all features by default
                        interaction_values_temp = shap.TreeExplainer(model).shap_interaction_values(
                            X_val_array[interaction_sample_idx]
                        )
                        
                        if isinstance(interaction_values_temp, list):
                            interaction_values_temp = interaction_values_temp[1]  # Binary classification
                        
                        # If we only want top features, extract subset
                        if len(interaction_indices) < len(feature_names):
                            # Extract only the interaction values for selected features
                            interaction_values_subset = interaction_values_temp[:, 
                                                                              np.ix_(interaction_indices, interaction_indices)]
                            interaction_values_temp = interaction_values_subset
                            selected_feature_names = top_interaction_features
                        else:
                            selected_feature_names = feature_names
                        
                        # Store in function-level variable for return and later use
                        interaction_values_result = interaction_values_temp
                        interaction_sample_indices_result = interaction_sample_idx
                        top_interaction_features_selected = selected_feature_names  # Store for later use
                        interaction_values = interaction_values_temp  # For use in this section
                        
                        logger.info(f"  ✓ Interaction values computed: shape {interaction_values.shape}")
                        
                        # Plot interaction summary only if generate_interaction is True
                        if generate_interaction:
                            plt.figure(figsize=(12, 10))
                            shap.summary_plot(
                                interaction_values,
                                X_val_array[interaction_sample_idx],
                                feature_names=feature_names,
                                max_display=max_display,
                                plot_type="compact_dot",
                                show=False
                            )
                            
                            interaction_path = os.path.join(output_dir, 'shap_interaction_summary.png')
                            plt.savefig(interaction_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"  ✓ Interaction plot saved: {interaction_path}")
                        
                except Exception as e:
                    logger.warning(f"  ⚠ Error computing interaction values: {e}")
                    logger.info("    Note: Interaction values computation can be slow for large datasets")
                    interaction_values_result = None
                    interaction_sample_indices_result = None
            
            # 7. Interaction Dependence Plot
            # Interaction Dependence Plot shows how two features interact with each other
            # It visualizes the joint effect of two features on the model output
            # 
            # How to interpret:
            # - X-axis: Primary feature value
            # - Y-axis: SHAP value for the primary feature
            # - Color: Secondary feature value (shows interaction effect)
            # - If color varies vertically at same x-value, there's an interaction
            # - Strong interaction: SHAP value depends on both features together
            # 
            # 交互依赖性图显示两个特征如何相互交互
            # 它可视化两个特征对模型输出的联合效应
            # 
            # 如何解读：
            # - X轴：主要特征值
            # - Y轴：主要特征的SHAP值
            # - 颜色：次要特征值（显示交互效应）
            # - 如果在相同x值处颜色垂直变化，则存在交互
            # - 强交互：SHAP值同时依赖于两个特征
            if generate_interaction_dependence and num_interaction_dependence_features > 0:
                logger.info(f"  Generating interaction dependence plots for top {num_interaction_dependence_features} features...")
                top_features = feature_importance.head(num_interaction_dependence_features)['feature'].tolist()
                
                for idx, primary_feature in enumerate(top_features):
                    try:
                        if primary_feature not in feature_names:
                            continue
                        
                        primary_feature_idx = feature_names.index(primary_feature)
                        
                        # Find the feature with strongest interaction with primary feature
                        # We'll use correlation between primary feature SHAP values and other features
                        # to find potential interaction partners
                        primary_shap = shap_values[:, primary_feature_idx]
                        
                        # Calculate potential interaction strength with other top features
                        interaction_scores = {}
                        for other_feature in top_features:
                            if other_feature == primary_feature:
                                continue
                            if other_feature not in feature_names:
                                continue
                            
                            other_feature_idx = feature_names.index(other_feature)
                            other_feature_values = X_val_array[:, other_feature_idx]
                            
                            # Calculate interaction score based on variance explained
                            # Higher variance in SHAP values when conditioning on other feature = stronger interaction
                            try:
                                # Use correlation between primary SHAP and other feature values
                                # Also consider how SHAP values vary with other feature
                                correlation = np.abs(np.corrcoef(primary_shap, other_feature_values)[0, 1])
                                if not np.isnan(correlation):
                                    interaction_scores[other_feature] = correlation
                            except:
                                pass
                        
                        # Select interaction feature
                        interaction_feature = None
                        interaction_feature_idx = None
                        
                        if interaction_scores:
                            # Get feature with highest interaction score
                            interaction_feature = max(interaction_scores, key=interaction_scores.get)
                            interaction_feature_idx = feature_names.index(interaction_feature)
                        else:
                            # Fallback: use second most important feature
                            if len(top_features) > 1:
                                for other_feature in top_features:
                                    if other_feature != primary_feature and other_feature in feature_names:
                                        interaction_feature = other_feature
                                        interaction_feature_idx = feature_names.index(other_feature)
                                        break
                        
                        if interaction_feature is None or interaction_feature_idx is None:
                            logger.debug(f"    Skipping {primary_feature}: no suitable interaction partner found")
                            continue
                        
                        # Generate interaction dependence plot
                        plt.figure(figsize=(12, 8))
                        
                        # Try using shap.plots.scatter (newer API) for better visualization
                        try:
                            # Create Explanation object
                            explanation = shap.Explanation(
                                values=shap_values,
                                base_values=np.full(len(shap_values), base_value),
                                data=X_val_array,
                                feature_names=feature_names
                            )
                            
                            # Use scatter plot with interaction
                            shap.plots.scatter(
                                explanation[:, primary_feature_idx],
                                color=explanation[:, interaction_feature_idx],
                                show=False
                            )
                            
                            plt.title(
                                f'Interaction Dependence Plot: {primary_feature} × {interaction_feature}\n'
                                f'(X-axis: {primary_feature}, Y-axis: SHAP({primary_feature}), Color: {interaction_feature})',
                                fontsize=12, pad=15, fontweight='bold'
                            )
                            plt.xlabel(f'{primary_feature} (feature value)', fontsize=11, fontweight='bold')
                            plt.ylabel(f'SHAP value for {primary_feature}', fontsize=11, fontweight='bold')
                            
                        except Exception as e_scatter:
                            logger.debug(f"    shap.plots.scatter failed: {e_scatter}")
                            # Fallback to enhanced dependence_plot
                            try:
                                shap.dependence_plot(
                                    primary_feature_idx,
                                    shap_values,
                                    X_val_array,
                                    feature_names=feature_names,
                                    interaction_index=interaction_feature_idx,
                                    show=False,
                                    alpha=0.6,
                                    dot_size=16
                                )
                                plt.title(
                                    f'Interaction Dependence Plot: {primary_feature} × {interaction_feature}\n'
                                    f'(X-axis: {primary_feature}, Y-axis: SHAP({primary_feature}), Color: {interaction_feature})',
                                    fontsize=12, pad=15, fontweight='bold'
                                )
                                plt.xlabel(f'{primary_feature} (feature value)', fontsize=11, fontweight='bold')
                                plt.ylabel(f'SHAP value for {primary_feature}', fontsize=11, fontweight='bold')
                            except Exception as e_dep:
                                logger.debug(f"    Enhanced dependence_plot also failed: {e_dep}")
                                # Last resort: manual scatter plot
                                try:
                                    primary_values = X_val_array[:, primary_feature_idx]
                                    interaction_values = X_val_array[:, interaction_feature_idx]
                                    
                                    scatter = plt.scatter(
                                        primary_values,
                                        primary_shap,
                                        c=interaction_values,
                                        cmap='viridis',
                                        alpha=0.6,
                                        s=20,
                                        edgecolors='black',
                                        linewidths=0.3
                                    )
                                    plt.colorbar(scatter, label=interaction_feature, pad=0.02)
                                    plt.xlabel(f'{primary_feature} (feature value)', fontsize=11, fontweight='bold')
                                    plt.ylabel(f'SHAP value for {primary_feature}', fontsize=11, fontweight='bold')
                                    plt.title(
                                        f'Interaction Dependence Plot: {primary_feature} × {interaction_feature}\n'
                                        f'(Color represents {interaction_feature} value)',
                                        fontsize=12, pad=15, fontweight='bold'
                                    )
                                    plt.grid(True, alpha=0.3, linestyle='--')
                                except Exception as e_manual:
                                    logger.warning(f"    All methods failed for {primary_feature}: {e_manual}")
                                    plt.close()
                                    continue
                        
                        # Save plot - sanitize feature names first
                        safe_primary = str(primary_feature).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
                        safe_interaction = str(interaction_feature).replace(' ', '_').replace('/', '_').replace('\\', '_').replace('|', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace('.', '_')
                        interaction_dep_path = os.path.join(
                            output_dir, 
                            f'shap_interaction_dependence_{safe_primary}_x_{safe_interaction}.png'
                        )
                        
                        plt.tight_layout()
                        plt.savefig(interaction_dep_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(
                            f"    ✓ Interaction dependence plot {idx+1}/{len(top_features)}: "
                            f"{primary_feature} × {interaction_feature} saved: {interaction_dep_path}"
                        )
                        
                    except Exception as e:
                        logger.warning(f"    ⚠ Error generating interaction dependence plot for {primary_feature}: {e}")
                        import traceback
                        logger.debug(f"    Traceback: {traceback.format_exc()}")
                        try:
                            plt.close()
                        except:
                            pass
            
            # 8. Shapley Interaction Index
            # Shapley Interaction Index quantifies the interaction strength between feature pairs
            # It is computed as the mean absolute interaction value between two features
            # 
            # The interaction index matrix is symmetric: I(i,j) = I(j,i)
            # Higher values indicate stronger interactions between features
            # 
            # Shapley交互指数量化特征对之间的交互强度
            # 它计算为两个特征之间的平均绝对交互值
            # 
            # 交互指数矩阵是对称的：I(i,j) = I(j,i)
            # 较高的值表示特征之间更强的交互
            if generate_shapley_interaction_index and interaction_values_result is not None:
                interaction_values = interaction_values_result  # Use stored values
                interaction_sample_idx = interaction_sample_indices_result
                logger.info("  Computing Shapley Interaction Index...")
                try:
                    # interaction_values shape: (n_samples, n_features, n_features)
                    # interaction_values[i, j, k] = interaction between features j and k for sample i
                    # For diagonal elements (j == k), it represents the main effect
                    # For off-diagonal elements (j != k), it represents the interaction effect
                    
                    n_samples, n_features, _ = interaction_values.shape
                    logger.info(f"    Interaction values shape: {interaction_values.shape}")
                    
                    # Compute Shapley Interaction Index as mean absolute interaction value
                    # I(i,j) = mean(|interaction_values[:, i, j]|)
                    # This gives us a symmetric matrix of interaction strengths
                    interaction_index_matrix = np.zeros((n_features, n_features))
                    
                    for i in range(n_features):
                        for j in range(n_features):
                            # Interaction index is the mean absolute value
                            # For symmetric interactions, we can also use (i,j) + (j,i) / 2
                            # But SHAP interaction values are already symmetric, so just use one
                            interaction_index_matrix[i, j] = np.abs(interaction_values[:, i, j]).mean()
                    
                    # Create feature names for the features used in interaction computation
                    # n_features is the number of features in interaction_values
                    # We need to map back to original feature names
                    if top_interaction_features_selected is not None:
                        top_interaction_features = top_interaction_features_selected
                    elif interaction_sample_idx is not None:
                        # Fallback: use top n_features from feature importance
                        top_interaction_features = feature_importance.head(n_features)['feature'].tolist()
                    else:
                        top_interaction_features = feature_names[:n_features]
                    
                    # Create DataFrame for interaction index matrix
                    interaction_index_df = pd.DataFrame(
                        interaction_index_matrix,
                        index=top_interaction_features,
                        columns=top_interaction_features
                    )
                    
                    # Extract top interaction pairs (excluding diagonal)
                    interaction_pairs = []
                    for i in range(n_features):
                        for j in range(i + 1, n_features):  # Only upper triangle to avoid duplicates
                            pair_name = f"{top_interaction_features[i]} × {top_interaction_features[j]}"
                            interaction_pairs.append({
                                'feature_1': top_interaction_features[i],
                                'feature_2': top_interaction_features[j],
                                'interaction_index': interaction_index_matrix[i, j],
                                'abs_interaction_index': np.abs(interaction_index_matrix[i, j])
                            })
                    
                    interaction_pairs_df = pd.DataFrame(interaction_pairs)
                    interaction_pairs_df = interaction_pairs_df.sort_values('abs_interaction_index', ascending=False)
                    
                    logger.info(f"\n    Top 10 Feature Interactions (by Shapley Interaction Index):")
                    for idx, row in interaction_pairs_df.head(10).iterrows():
                        logger.info(f"      {row['feature_1']} × {row['feature_2']}: {row['interaction_index']:.6f}")
                    
                    # Save interaction index matrix and top pairs to CSV
                    interaction_matrix_path = os.path.join(output_dir, 'shapley_interaction_index_matrix.csv')
                    interaction_index_df.to_csv(interaction_matrix_path)
                    logger.info(f"    ✓ Interaction index matrix saved: {interaction_matrix_path}")
                    
                    interaction_pairs_path = os.path.join(output_dir, 'shapley_interaction_index_pairs.csv')
                    interaction_pairs_df.to_csv(interaction_pairs_path, index=False)
                    logger.info(f"    ✓ Interaction pairs saved: {interaction_pairs_path}")
                    
                    # Store for return value
                    interaction_index_matrix_result = interaction_index_matrix
                    interaction_pairs_result = interaction_pairs_df
                    
                    # Visualize interaction index matrix as heatmap
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            # Determine number of features to display in heatmap
                            max_features_heatmap = min(20, n_features)  # Limit for readability
                            if n_features > max_features_heatmap:
                                # Use top features by interaction strength
                                top_features_by_interaction = []
                                feature_interaction_sums = interaction_index_matrix.sum(axis=1) - np.diag(interaction_index_matrix)
                                top_indices = np.argsort(feature_interaction_sums)[-max_features_heatmap:]
                                top_indices = sorted(top_indices)
                                
                                interaction_matrix_plot = interaction_index_matrix[np.ix_(top_indices, top_indices)]
                                feature_names_plot = [top_interaction_features[i] for i in top_indices]
                            else:
                                interaction_matrix_plot = interaction_index_matrix
                                feature_names_plot = top_interaction_features
                            
                            # Create heatmap
                            fig, ax = plt.subplots(figsize=(max(12, len(feature_names_plot) * 0.8), 
                                                              max(10, len(feature_names_plot) * 0.8)))
                            
                            # Use seaborn if available for better visualization
                            if SEABORN_AVAILABLE:
                                sns.heatmap(
                                    interaction_matrix_plot,
                                    xticklabels=feature_names_plot,
                                    yticklabels=feature_names_plot,
                                    annot=True,
                                    fmt='.4f',
                                    cmap='YlOrRd',
                                    square=True,
                                    linewidths=0.5,
                                    cbar_kws={'label': 'Shapley Interaction Index'},
                                    ax=ax,
                                    annot_kws={'size': 8}
                                )
                            else:
                                # Fallback to matplotlib
                                im = ax.imshow(interaction_matrix_plot, cmap='YlOrRd', aspect='auto')
                                ax.set_xticks(np.arange(len(feature_names_plot)))
                                ax.set_yticks(np.arange(len(feature_names_plot)))
                                ax.set_xticklabels(feature_names_plot, rotation=45, ha='right', fontsize=8)
                                ax.set_yticklabels(feature_names_plot, fontsize=8)
                                
                                # Add text annotations
                                for i in range(len(feature_names_plot)):
                                    for j in range(len(feature_names_plot)):
                                        text = ax.text(j, i, f'{interaction_matrix_plot[i, j]:.3f}',
                                                      ha="center", va="center", color="black", fontsize=6)
                                
                                plt.colorbar(im, ax=ax, label='Shapley Interaction Index')
                            
                            ax.set_title('Shapley Interaction Index Matrix\n'
                                       '(Higher values indicate stronger feature interactions)',
                                       fontsize=14, pad=15, fontweight='bold')
                            ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
                            
                            plt.tight_layout()
                            
                            heatmap_path = os.path.join(output_dir, 'shapley_interaction_index_heatmap.png')
                            plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"    ✓ Interaction index heatmap saved: {heatmap_path}")
                            
                        except Exception as e_heatmap:
                            logger.warning(f"    ⚠ Error generating interaction index heatmap: {e_heatmap}")
                            import traceback
                            logger.debug(f"    Traceback: {traceback.format_exc()}")
                            try:
                                plt.close()
                            except:
                                pass
                    
                except Exception as e:
                    logger.warning(f"  ⚠ Error computing Shapley Interaction Index: {e}")
                    import traceback
                    logger.debug(f"  Full traceback: {traceback.format_exc()}")
            
            # 9. SHAP Clustering
            # SHAP Clustering identifies heterogeneous subgroups with different feature patterns
            # This helps discover that samples cluster into distinct groups with different driving features
            # 
            # Use case: If clustering reveals clear subgroups (e.g., old users vs new users),
            # consider training separate models for each subgroup and ensembling them
            # 
            # SHAP聚类识别具有不同特征模式的异质性子群体
            # 这有助于发现样本被分成具有不同驱动特征的明显组别
            # 
            # 应用场景：如果聚类显示明显的子组（例如，老用户 vs 新用户），
            # 考虑为每个子组训练单独的模型并集成它们
            if generate_shap_clustering:
                logger.info("  Performing SHAP-based clustering analysis...")
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    # Use SHAP values as features for clustering
                    # Each sample is represented by its SHAP values across all features
                    logger.info(f"    Clustering {len(shap_values)} samples into {num_clusters} clusters based on SHAP values...")
                    
                    # Standardize SHAP values for clustering
                    scaler = StandardScaler()
                    shap_values_scaled = scaler.fit_transform(shap_values)
                    
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(shap_values_scaled)
                    
                    # Analyze feature importance per cluster
                    cluster_feature_importance = {}
                    for cluster_id in range(num_clusters):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_shap = shap_values[cluster_mask]
                        
                        # Calculate mean absolute SHAP value per feature for this cluster
                        cluster_mean_abs_shap = np.abs(cluster_shap).mean(axis=0)
                        cluster_feature_importance[cluster_id] = cluster_mean_abs_shap
                        
                        logger.info(f"\n    Cluster {cluster_id} (n={cluster_mask.sum()}):")
                        logger.info(f"      Top 5 features:")
                        cluster_importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'mean_abs_shap': cluster_mean_abs_shap
                        }).sort_values('mean_abs_shap', ascending=False)
                        
                        for idx, row in cluster_importance_df.head(5).iterrows():
                            logger.info(f"        {row['feature']}: {row['mean_abs_shap']:.4f}")
                    
                    # Save cluster assignments
                    cluster_assignments_df = pd.DataFrame({
                        'sample_index': np.arange(len(shap_values)),
                        'cluster': cluster_labels
                    })
                    
                    # Add original data indices if available
                    if sample_indices is not None:
                        cluster_assignments_df['original_index'] = sample_indices
                    
                    cluster_assignments_path = os.path.join(output_dir, 'shap_cluster_assignments.csv')
                    cluster_assignments_df.to_csv(cluster_assignments_path, index=False)
                    logger.info(f"\n    ✓ Cluster assignments saved: {cluster_assignments_path}")
                    
                    # Store for return value
                    cluster_labels_result = cluster_labels
                    
                    # Save cluster-specific feature importance
                    cluster_importance_all = pd.DataFrame(cluster_feature_importance, index=feature_names)
                    cluster_importance_all.columns = [f'cluster_{i}' for i in range(num_clusters)]
                    cluster_importance_path = os.path.join(output_dir, 'shap_cluster_feature_importance.csv')
                    cluster_importance_all.to_csv(cluster_importance_path)
                    logger.info(f"    ✓ Cluster feature importance saved: {cluster_importance_path}")
                    
                    # Visualize clusters
                    if MATPLOTLIB_AVAILABLE:
                        try:
                            # Use PCA to reduce dimensionality for visualization
                            from sklearn.decomposition import PCA
                            
                            # Reduce SHAP values to 2D for visualization
                            pca = PCA(n_components=2, random_state=42)
                            shap_2d = pca.fit_transform(shap_values_scaled)
                            
                            # Create scatter plot colored by cluster
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            if SEABORN_AVAILABLE:
                                # Use seaborn for better color palette
                                scatter = ax.scatter(shap_2d[:, 0], shap_2d[:, 1], 
                                                    c=cluster_labels, cmap='viridis', 
                                                    alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
                                plt.colorbar(scatter, ax=ax, label='Cluster')
                            else:
                                # Use matplotlib color map
                                scatter = ax.scatter(shap_2d[:, 0], shap_2d[:, 1], 
                                                    c=cluster_labels, cmap='viridis', 
                                                    alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
                                plt.colorbar(scatter, ax=ax, label='Cluster')
                            
                            # Plot cluster centers
                            centers_2d = pca.transform(kmeans.cluster_centers_)
                            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                                      c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
                            
                            ax.set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})', 
                                         fontsize=11, fontweight='bold')
                            ax.set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})', 
                                         fontsize=11, fontweight='bold')
                            ax.set_title(f'SHAP-based Clustering Visualization ({num_clusters} clusters)\n'
                                       'Each point represents a sample; color indicates cluster membership',
                                       fontsize=12, pad=15, fontweight='bold')
                            ax.legend()
                            ax.grid(True, alpha=0.3, linestyle='--')
                            
                            plt.tight_layout()
                            cluster_plot_path = os.path.join(output_dir, 'shap_clustering_visualization.png')
                            plt.savefig(cluster_plot_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"    ✓ Clustering visualization saved: {cluster_plot_path}")
                            
                            # Create heatmap showing feature importance per cluster
                            fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.3), max(8, num_clusters * 0.8)))
                            
                            # Select top features for visualization
                            top_features_for_cluster = min(20, len(feature_names))
                            top_feature_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-top_features_for_cluster:]
                            top_feature_names = [feature_names[i] for i in top_feature_indices]
                            
                            cluster_importance_plot = cluster_importance_all.loc[top_feature_names].values
                            
                            if SEABORN_AVAILABLE:
                                sns.heatmap(cluster_importance_plot, 
                                          xticklabels=[f'Cluster {i}' for i in range(num_clusters)],
                                          yticklabels=top_feature_names,
                                          annot=True, fmt='.3f', cmap='YlOrRd',
                                          cbar_kws={'label': 'Mean |SHAP value|'},
                                          ax=ax)
                            else:
                                im = ax.imshow(cluster_importance_plot, cmap='YlOrRd', aspect='auto')
                                ax.set_xticks(np.arange(num_clusters))
                                ax.set_yticks(np.arange(len(top_feature_names)))
                                ax.set_xticklabels([f'Cluster {i}' for i in range(num_clusters)])
                                ax.set_yticklabels(top_feature_names)
                                plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
                                
                                # Add text annotations
                                for i in range(len(top_feature_names)):
                                    for j in range(num_clusters):
                                        text = ax.text(j, i, f'{cluster_importance_plot[i, j]:.3f}',
                                                      ha="center", va="center", color="black", fontsize=8)
                            
                            ax.set_title('Feature Importance per Cluster\n'
                                       '(Higher values indicate stronger feature influence in that cluster)',
                                       fontsize=12, pad=15, fontweight='bold')
                            ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
                            ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
                            
                            plt.tight_layout()
                            cluster_heatmap_path = os.path.join(output_dir, 'shap_cluster_feature_importance_heatmap.png')
                            plt.savefig(cluster_heatmap_path, bbox_inches='tight', dpi=150)
                            plt.close()
                            logger.info(f"    ✓ Cluster feature importance heatmap saved: {cluster_heatmap_path}")
                            
                        except Exception as e_cluster_viz:
                            logger.warning(f"    ⚠ Error generating clustering visualizations: {e_cluster_viz}")
                            import traceback
                            logger.debug(f"    Traceback: {traceback.format_exc()}")
                            try:
                                plt.close()
                            except:
                                pass
                    
                except ImportError as e_import:
                    logger.warning(f"  ⚠ Error: sklearn not available for clustering: {e_import}")
                    logger.info("    Install sklearn to enable SHAP clustering: pip install scikit-learn")
                except Exception as e:
                    logger.warning(f"  ⚠ Error performing SHAP clustering: {e}")
                    import traceback
                    logger.debug(f"  Full traceback: {traceback.format_exc()}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Error generating plots: {e}")
            import traceback
            logger.debug(f"  Full error: {traceback.format_exc()}")
    
    # Save feature importance to CSV
    importance_csv_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_csv_path, index=False)
    logger.info(f"  ✓ Feature importance saved: {importance_csv_path}")
    
    logger.info(f"{'='*60}")
    
    result = {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'explainer': explainer,
        'sample_indices': sample_indices,
        'base_value': base_value,
        'X_val_array': X_val_array,
        'feature_names': feature_names
    }
    
    # Add interaction-related results if computed
    if interaction_values_result is not None:
        result['interaction_values'] = interaction_values_result
        if interaction_sample_indices_result is not None:
            result['interaction_sample_indices'] = interaction_sample_indices_result
    
    if interaction_index_matrix_result is not None:
        result['interaction_index_matrix'] = interaction_index_matrix_result
    if interaction_pairs_result is not None:
        result['interaction_pairs'] = interaction_pairs_result
    if cluster_labels_result is not None:
        result['cluster_labels'] = cluster_labels_result
    
    # Analyze misclassified samples if requested
    if analyze_errors:
        if y_true is None or y_pred is None:
            logger.warning("  ⚠ analyze_errors=True but y_true or y_pred not provided. Skipping error analysis.")
        else:
            try:
                logger.info("\n" + "="*60)
                logger.info("Analyzing Misclassified Samples")
                logger.info("="*60)
                
                # Convert X_val back to DataFrame if needed for error analysis
                if isinstance(X_val, pd.DataFrame):
                    X_for_errors = X_val
                else:
                    X_for_errors = pd.DataFrame(X_val_array, columns=feature_names)
                
                logger.info(f"  X_for_errors shape: {X_for_errors.shape}")
                logger.info(f"  y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
                logger.info(f"  sample_indices: {sample_indices is not None}")
                
                # Adjust indices if sampling was applied in analyze_with_shap
                # If y_true/y_pred length matches X_for_errors, they are already aligned (from generate_shap_visualization.py)
                # Otherwise, we need to apply sample_indices
                if sample_indices is not None and len(y_true) > len(X_for_errors):
                    # y_true/y_pred are from original data, need to apply sampling
                    logger.info(f"  Applying sample_indices to y_true/y_pred (original length: {len(y_true)})")
                    y_true_adjusted = y_true[sample_indices] if len(y_true) > max(sample_indices) else y_true
                    y_pred_adjusted = y_pred[sample_indices] if len(y_pred) > max(sample_indices) else y_pred
                elif len(y_true) == len(X_for_errors) and len(y_pred) == len(X_for_errors):
                    # y_true/y_pred are already aligned with X_for_errors (already sampled)
                    logger.info("  y_true/y_pred already aligned with X_for_errors")
                    y_true_adjusted = y_true
                    y_pred_adjusted = y_pred
                else:
                    # Lengths don't match, try to align
                    logger.warning(f"  ⚠ Length mismatch: X_for_errors={len(X_for_errors)}, y_true={len(y_true)}, y_pred={len(y_pred)}")
                    logger.info("  Attempting to align by truncating to minimum length")
                    y_true_adjusted = y_true
                    y_pred_adjusted = y_pred
                
                # Ensure lengths match
                min_len = min(len(X_for_errors), len(y_true_adjusted), len(y_pred_adjusted))
                if min_len < len(X_for_errors):
                    logger.warning(f"  ⚠ Truncating to {min_len} samples due to length mismatch")
                X_for_errors = X_for_errors.iloc[:min_len]
                y_true_adjusted = y_true_adjusted[:min_len]
                y_pred_adjusted = y_pred_adjusted[:min_len]
                
                logger.info(f"  Final aligned lengths: X={len(X_for_errors)}, y_true={len(y_true_adjusted)}, y_pred={len(y_pred_adjusted)}")
                
                # Create error output directory
                error_output_dir = os.path.join(output_dir, 'errors')
                
                error_analysis_result = analyze_misclassified_samples(
                    model=model,
                    X=X_for_errors,
                    y_true=y_true_adjusted,
                    y_pred=y_pred_adjusted,
                    feature_names=feature_names,
                    output_dir=error_output_dir,
                    max_error_force_plots=max_error_force_plots,
                    max_display=max_display
                )
                
                result['error_analysis'] = error_analysis_result
                logger.info("  ✓ Error analysis complete")
                
            except Exception as e:
                logger.warning(f"  ⚠ Error during misclassified samples analysis: {e}")
                import traceback
                logger.debug(f"  Full error: {traceback.format_exc()}")
                result['error_analysis'] = None
    
    return result


def _generate_force_plot_for_sample(sample_idx: int, shap_values: np.ndarray, X_val_array: np.ndarray,
                                    feature_names: List[str], base_value: float, output_path: str,
                                    max_display: int = 20) -> bool:
    """
    Generate force plot for a single sample.
    为单个样本生成力图。
    
    Args:
        sample_idx: Index of the sample
        shap_values: SHAP values array
        X_val_array: Feature values array
        feature_names: List of feature names
        base_value: Base value from explainer
        output_path: Path to save the force plot
        max_display: Maximum number of features to display
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sample_shap = shap_values[sample_idx]
        sample_data = X_val_array[sample_idx]
        
        # Try Method 1: SHAP force plot with matplotlib
        try:
            fig = plt.figure(figsize=(12, 6))
            shap.plots.force(
                base_value,
                sample_shap,
                sample_data,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
                fig=fig
            )
            # Add sample index annotation
            fig.text(0.98, 0.98, f'Sample: {sample_idx}', 
                    transform=fig.transFigure,
                    fontsize=10, fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            return True
        except Exception:
            try:
                plt.close()
            except:
                pass
        
        # Method 2: Manual horizontal bar plot
        try:
            fig, ax = plt.subplots(figsize=(14, max(6, max_display * 0.4)))
            
            # Sort features by absolute SHAP value
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sample_shap,
                'feature_value': sample_data,
                'abs_shap': np.abs(sample_shap)
            }).sort_values('abs_shap', ascending=False).head(max_display)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(feature_df))
            colors = ['red' if x > 0 else 'blue' for x in feature_df['shap_value'].values]
            
            bars = ax.barh(y_pos, feature_df['shap_value'].values, 
                          align='center', color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add feature values to labels
            labels = []
            for fname, val, shap_val in zip(feature_df['feature'].values, 
                                           feature_df['feature_value'].values,
                                           feature_df['shap_value'].values):
                if isinstance(val, (int, float)):
                    if abs(val) < 0.01:
                        val_str = f"{val:.4f}"
                    elif abs(val) < 1:
                        val_str = f"{val:.3f}"
                    else:
                        val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                labels.append(f"{fname}\n(val={val_str}, SHAP={shap_val:.3f})")
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('SHAP Value (contribution to model output)', fontsize=10, fontweight='bold')
            final_pred = base_value + sample_shap.sum()
            ax.set_title(f'Force Plot - Sample {sample_idx}\nBase value: {base_value:.4f} → Final prediction: {final_pred:.4f}', 
                        fontsize=11, pad=10, fontweight='bold')
            
            ax.text(base_value, len(feature_df) - 0.5, f'Base\n{base_value:.4f}', 
                   ha='center', va='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            
            # Add sample index annotation
            ax.text(0.98, 0.98, f'Sample: {sample_idx}', 
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            return True
        except Exception:
            try:
                plt.close()
            except:
                pass
        
        # Method 3: Simple bar chart as fallback
        try:
            fig, ax = plt.subplots(figsize=(12, max(6, max_display * 0.4)))
            feature_importance_sample = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sample_shap,
                'abs_shap': np.abs(sample_shap)
            }).sort_values('abs_shap', ascending=False).head(max_display)
            
            colors = ['red' if x > 0 else 'blue' for x in feature_importance_sample['shap_value'].values]
            y_pos = np.arange(len(feature_importance_sample))
            bars = ax.barh(y_pos, feature_importance_sample['shap_value'].values, 
                          align='center', color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_importance_sample['feature'].values)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('SHAP value (impact on model output)')
            final_pred = base_value + sample_shap.sum()
            ax.set_title(f'Force Plot - Sample {sample_idx}\nBase: {base_value:.4f} → Final: {final_pred:.4f}')
            # Add sample index annotation
            ax.text(0.98, 0.98, f'Sample: {sample_idx}', 
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            return True
        except Exception:
            try:
                plt.close()
            except:
                pass
        
        return False
    except Exception as e:
        logger.warning(f"Error generating force plot for sample {sample_idx}: {e}")
        try:
            plt.close()
        except:
            pass
        return False


def analyze_misclassified_samples(model: Any, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 output_dir: Optional[str] = None,
                                 max_error_force_plots: int = 10,
                                 max_display: int = 20) -> Dict[str, Any]:
    """
    Analyze misclassified samples using SHAP values.
    使用SHAP值分析预测错误的样本。
    
    Args:
        model: Trained XGBoost model
        X: Feature DataFrame (can be training or validation set)
        y_true: True labels array
        y_pred: Predicted labels array
        feature_names: List of feature names (if None, uses DataFrame columns)
        output_dir: Directory to save results (default: output/shap/errors/)
        max_error_force_plots: Maximum number of error samples to generate force plots for
        max_display: Maximum number of features to display in plots
        
    Returns:
        Dictionary containing:
        - 'summary': Summary statistics
        - 'misclassified_samples': List of misclassified sample details
        - 'common_patterns': Common patterns in misclassified samples
    """
    if not SHAP_AVAILABLE or shap is None:
        error_msg = "SHAP is not available. "
        if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
            error_msg += "SHAP may be installed in /Data/yuguang.yao/pytorch_packages but could not be imported. "
        error_msg += "Please ensure SHAP is properly installed: pip install shap"
        raise ImportError(error_msg)
    
    # Use unified path configuration if output_dir is not specified
    if output_dir is None:
        from src.utils.paths import get_shap_dir
        output_dir = str(get_shap_dir() / 'errors')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("SHAP Analysis: Misclassified Samples")
    logger.info("="*60)
    
    # Convert to numpy array
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_array = X
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
    
    # Ensure arrays are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Identify misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    total_samples = len(y_true)
    misclassified_count = len(misclassified_indices)
    error_rate = misclassified_count / total_samples if total_samples > 0 else 0.0
    
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Misclassified samples: {misclassified_count} ({error_rate:.2%})")
    
    if misclassified_count == 0:
        logger.info("  No misclassified samples found!")
        return {
            'summary': {
                'total_samples': total_samples,
                'misclassified_count': 0,
                'error_rate': 0.0,
                'error_types': {'false_positive': 0, 'false_negative': 0}
            },
            'misclassified_samples': [],
            'common_patterns': {}
        }
    
    # Classify error types
    false_positive_mask = (y_true[misclassified_indices] == 0) & (y_pred[misclassified_indices] == 1)
    false_negative_mask = (y_true[misclassified_indices] == 1) & (y_pred[misclassified_indices] == 0)
    
    false_positive_count = np.sum(false_positive_mask)
    false_negative_count = np.sum(false_negative_mask)
    
    logger.info(f"  False positives (predicted 1, actual 0): {false_positive_count}")
    logger.info(f"  False negatives (predicted 0, actual 1): {false_negative_count}")
    
    # Calculate SHAP values for misclassified samples only
    logger.info("  Computing SHAP values for misclassified samples...")
    explainer = shap.TreeExplainer(model)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    
    X_misclassified = X_array[misclassified_indices]
    shap_values_misclassified = explainer.shap_values(X_misclassified)
    
    # Handle binary classification
    if isinstance(shap_values_misclassified, list):
        shap_values_misclassified = shap_values_misclassified[1]
    
    logger.info(f"  ✓ SHAP values computed for {len(misclassified_indices)} misclassified samples")
    logger.info(f"  Base value: {base_value:.4f}")
    
    # Calculate feature importance for misclassified samples
    mean_abs_shap_misclassified = np.abs(shap_values_misclassified).mean(axis=0)
    feature_importance_misclassified = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap_misclassified
    }).sort_values('mean_abs_shap', ascending=False)
    
    logger.info(f"\n  Top {min(max_display, len(feature_importance_misclassified))} features in misclassified samples:")
    for idx, row in feature_importance_misclassified.head(max_display).iterrows():
        logger.info(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Prepare detailed sample information
    misclassified_samples_list = []
    force_plot_paths = []
    
    # Select samples for force plot generation (prioritize by SHAP magnitude or random)
    num_force_plots = min(max_error_force_plots, misclassified_count)
    
    # Calculate total SHAP magnitude for each misclassified sample
    shap_magnitudes = np.abs(shap_values_misclassified).sum(axis=1)
    # Select top samples by SHAP magnitude
    top_indices_by_magnitude = np.argsort(shap_magnitudes)[-num_force_plots:][::-1]
    
    for idx, misclassified_idx in enumerate(misclassified_indices):
        sample_idx_in_misclassified = np.where(misclassified_indices == misclassified_idx)[0][0]
        
        true_label = int(y_true[misclassified_idx])
        predicted_label = int(y_pred[misclassified_idx])
        error_type = 'false_positive' if (true_label == 0 and predicted_label == 1) else 'false_negative'
        
        # Get feature values and SHAP values for this sample
        sample_features = X_array[misclassified_idx]
        sample_shap = shap_values_misclassified[sample_idx_in_misclassified]
        
        # Convert to dictionaries for JSON serialization
        features_dict = {feature_names[i]: float(sample_features[i]) if isinstance(sample_features[i], (int, float, np.number)) else str(sample_features[i])
                        for i in range(len(feature_names))}
        shap_dict = {feature_names[i]: float(sample_shap[i]) for i in range(len(feature_names))}
        
        sample_info = {
            'sample_index': int(misclassified_idx),
            'true_label': true_label,
            'predicted_label': predicted_label,
            'error_type': error_type,
            'features': features_dict,
            'shap_values': shap_dict,
            'shap_magnitude': float(shap_magnitudes[sample_idx_in_misclassified])
        }
        
        # Generate force plot for selected samples
        if sample_idx_in_misclassified in top_indices_by_magnitude:
            force_plot_filename = f'force_sample_{misclassified_idx}.png'
            force_plot_path = os.path.join(output_dir, force_plot_filename)
            
            if _generate_force_plot_for_sample(
                sample_idx_in_misclassified,
                shap_values_misclassified,
                X_misclassified,
                feature_names,
                base_value,
                force_plot_path,
                max_display
            ):
                sample_info['force_plot_path'] = force_plot_path
                force_plot_paths.append(force_plot_path)
                logger.info(f"    ✓ Force plot generated for sample {misclassified_idx}")
            else:
                logger.warning(f"    ⚠ Failed to generate force plot for sample {misclassified_idx}")
        
        misclassified_samples_list.append(sample_info)
    
    # Calculate common patterns
    # Top features that contribute most to misclassification
    top_misclassified_features = feature_importance_misclassified.head(20)['feature'].tolist()
    
    # Average feature values for false positives vs false negatives
    false_positive_indices = misclassified_indices[false_positive_mask]
    false_negative_indices = misclassified_indices[false_negative_mask]
    
    avg_features_fp = {}
    avg_features_fn = {}
    
    if len(false_positive_indices) > 0:
        fp_features = X_array[false_positive_indices]
        for i, feature_name in enumerate(feature_names):
            if np.issubdtype(fp_features[:, i].dtype, np.number):
                avg_features_fp[feature_name] = float(np.nanmean(fp_features[:, i]))
    
    if len(false_negative_indices) > 0:
        fn_features = X_array[false_negative_indices]
        for i, feature_name in enumerate(feature_names):
            if np.issubdtype(fn_features[:, i].dtype, np.number):
                avg_features_fn[feature_name] = float(np.nanmean(fn_features[:, i]))
    
    # Build result dictionary
    result = {
        'summary': {
            'total_samples': int(total_samples),
            'misclassified_count': int(misclassified_count),
            'error_rate': float(error_rate),
            'error_types': {
                'false_positive': int(false_positive_count),
                'false_negative': int(false_negative_count)
            },
            'base_value': float(base_value)
        },
        'misclassified_samples': misclassified_samples_list,
        'common_patterns': {
            'top_misclassified_features': top_misclassified_features,
            'average_features_false_positive': avg_features_fp,
            'average_features_false_negative': avg_features_fn,
            'feature_importance_misclassified': {
                row['feature']: float(row['mean_abs_shap'])
                for _, row in feature_importance_misclassified.iterrows()
            }
        }
    }
    
    # Save to JSON
    json_path = os.path.join(output_dir, 'misclassified_samples.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  ✓ Results saved to: {json_path}")
    logger.info(f"  ✓ Generated {len(force_plot_paths)} force plots")
    
    logger.info(f"{'='*60}")
    
    return result


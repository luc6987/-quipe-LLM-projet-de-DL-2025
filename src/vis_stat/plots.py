"""
Feature Visualization Module
特征可视化模块

This module provides comprehensive visualization and statistical analysis for:
1. Tabular features (distributions, correlations, missing values)
2. Text features (length distributions, word frequencies)
3. Model predictions (distributions, correlations)
4. Statistical reports (CSV format)

All visualizations are saved as PNG files, and statistics are saved as CSV files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, List, Dict, Union, Tuple
import warnings
from collections import Counter
import re
from src.utils.paths import get_preprocessing_dir

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

warnings.filterwarnings('ignore')


def _ensure_dir(directory: str):
    """Ensure directory exists, create if not."""
    os.makedirs(directory, exist_ok=True)


def _get_timestamp() -> str:
    """Get timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_visualization_dir() -> str:
    """Unified default directory for preprocessing visualizations."""
    return str(get_preprocessing_dir('visualizations'))


def _default_stats_dir() -> str:
    """Unified default directory for preprocessing statistics."""
    return str(get_preprocessing_dir('visualizations/stats'))


def visualize_tabular_features(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    label_col: Optional[str] = None,
    max_features: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, str]:
    """
    Visualize tabular features: distributions, correlations, missing values.
    
    Args:
        df: DataFrame with tabular features
        output_dir: Output directory for plots
        label_col: Optional label column name for stratified analysis
        max_features: Maximum number of features to visualize (for correlation)
        figsize: Figure size tuple
        
    Returns:
        Dictionary mapping plot type to file path
    """
    if output_dir is None:
        output_dir = _default_visualization_dir()
    _ensure_dir(output_dir)
    timestamp = _get_timestamp()
    saved_files = {}
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove label column if present
    if label_col and label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if label_col and label_col in categorical_cols:
        categorical_cols.remove(label_col)
    
    # 1. Numeric Features Distribution
    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols[:n_rows * n_cols]):
            ax = axes[idx]
            data = df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(f'{col}\n(n={len(data)}, missing={df[col].isna().sum()})')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(col)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f'tabular_numeric_distributions_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['numeric_distributions'] = filepath
        print(f"  ✓ Saved numeric distributions: {filename}")
    
    # 2. Numeric Features Boxplots
    if numeric_cols:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols[:n_rows * n_cols]):
            ax = axes[idx]
            data = df[col].dropna()
            if len(data) > 0:
                ax.boxplot(data, vert=True)
                ax.set_title(f'{col}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(col)
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f'tabular_numeric_boxplots_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['numeric_boxplots'] = filepath
        print(f"  ✓ Saved numeric boxplots: {filename}")
    
    # 3. Categorical Features Frequency
    if categorical_cols:
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(categorical_cols[:n_rows * n_cols]):
            ax = axes[idx]
            value_counts = df[col].value_counts().head(20)  # Top 20 categories
            if len(value_counts) > 0:
                value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'{col}\n(unique={df[col].nunique()})')
                ax.set_xlabel('Category')
                ax.set_ylabel('Frequency')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(col)
        
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f'tabular_categorical_frequency_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['categorical_frequency'] = filepath
        print(f"  ✓ Saved categorical frequency: {filename}")
    
    # 4. Correlation Heatmap (for numeric features)
    if len(numeric_cols) > 1:
        # Limit to max_features for readability
        cols_to_plot = numeric_cols[:max_features]
        corr_matrix = df[cols_to_plot].corr()
        
        fig, ax = plt.subplots(figsize=(max(12, len(cols_to_plot) * 0.5), max(10, len(cols_to_plot) * 0.5)))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        ax.set_title(f'Feature Correlation Heatmap\n({len(cols_to_plot)} features)')
        plt.tight_layout()
        filename = f'tabular_correlation_heatmap_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['correlation_heatmap'] = filepath
        print(f"  ✓ Saved correlation heatmap: {filename}")
    
    # 5. Missing Values Heatmap
    missing_data = df.isnull()
    if missing_data.any().any():
        # Calculate missing percentage
        missing_pct = (missing_data.sum() / len(df) * 100).sort_values(ascending=False)
        cols_with_missing = missing_pct[missing_pct > 0].index.tolist()
        
        if cols_with_missing:
            # Limit columns for readability
            cols_to_plot = cols_with_missing[:50]
            missing_subset = missing_data[cols_to_plot]
            
            fig, ax = plt.subplots(figsize=(max(12, len(cols_to_plot) * 0.3), 8))
            sns.heatmap(
                missing_subset.T,
                cbar=True,
                yticklabels=True,
                xticklabels=False,
                cmap='YlOrRd',
                ax=ax
            )
            ax.set_title(f'Missing Values Heatmap\n({len(cols_to_plot)} features with missing data)')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            filename = f'tabular_missing_values_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved_files['missing_values'] = filepath
            print(f"  ✓ Saved missing values heatmap: {filename}")
    
    # 6. Label Distribution (if label column provided)
    if label_col and label_col in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        label_counts = df[label_col].value_counts().sort_index()
        axes[0].bar(label_counts.index, label_counts.values, color='steelblue', edgecolor='black')
        axes[0].set_title(f'Label Distribution\n({label_col})')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Label Proportion')
        
        plt.tight_layout()
        filename = f'tabular_label_distribution_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['label_distribution'] = filepath
        print(f"  ✓ Saved label distribution: {filename}")
    
    return saved_files


def visualize_text_features(
    texts: Union[List[str], pd.Series],
    output_dir: Optional[str] = None,
    labels: Optional[Union[List, pd.Series, np.ndarray]] = None,
    max_words: int = 30,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, str]:
    """
    Visualize text features: length distributions, word frequencies.
    
    Args:
        texts: List or Series of text strings
        output_dir: Output directory for plots
        labels: Optional labels for stratified analysis
        max_words: Maximum number of words to show in word frequency plot
        figsize: Figure size tuple
        
    Returns:
        Dictionary mapping plot type to file path
    """
    if output_dir is None:
        output_dir = _default_visualization_dir()
    _ensure_dir(output_dir)
    timestamp = _get_timestamp()
    saved_files = {}
    
    # Convert to list if needed
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [str(t) if t is not None else '' for t in texts]
    
    # 1. Text Length Distribution (characters)
    char_lengths = [len(text) for text in texts]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Character length histogram
    axes[0, 0].hist(char_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title(f'Text Length Distribution (Characters)\n(mean={np.mean(char_lengths):.1f}, median={np.median(char_lengths):.1f})')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Character length boxplot
    axes[0, 1].boxplot(char_lengths, vert=True)
    axes[0, 1].set_title('Text Length Boxplot (Characters)')
    axes[0, 1].set_ylabel('Character Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Text Length Distribution (words)
    word_lengths = [len(text.split()) for text in texts]
    
    axes[1, 0].hist(word_lengths, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].set_title(f'Text Length Distribution (Words)\n(mean={np.mean(word_lengths):.1f}, median={np.median(word_lengths):.1f})')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Word length boxplot
    axes[1, 1].boxplot(word_lengths, vert=True)
    axes[1, 1].set_title('Text Length Boxplot (Words)')
    axes[1, 1].set_ylabel('Word Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'text_length_distributions_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    saved_files['length_distributions'] = filepath
    print(f"  ✓ Saved text length distributions: {filename}")
    
    # 3. Word Frequency (Top N words)
    all_words = []
    for text in texts:
        # Simple word tokenization (split by whitespace and punctuation)
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    if all_words:
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(max_words)
        
        words, counts = zip(*top_words) if top_words else ([], [])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(words)), counts, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {max_words} Most Frequent Words\n(total unique words: {len(word_freq)})')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        filename = f'text_word_frequency_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['word_frequency'] = filepath
        print(f"  ✓ Saved word frequency: {filename}")
    
    # 4. Text Length vs Label (if labels provided)
    if labels is not None:
        if isinstance(labels, (pd.Series, np.ndarray)):
            labels = labels.tolist()
        
        # Create DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'char_length': char_lengths,
            'word_length': word_lengths,
            'label': labels
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Character length by label
        for label in sorted(set(labels)):
            label_data = plot_df[plot_df['label'] == label]['char_length']
            axes[0].hist(label_data, bins=30, alpha=0.6, label=f'Label {label}', edgecolor='black')
        axes[0].set_title('Text Length (Characters) by Label')
        axes[0].set_xlabel('Character Count')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Word length by label
        for label in sorted(set(labels)):
            label_data = plot_df[plot_df['label'] == label]['word_length']
            axes[1].hist(label_data, bins=30, alpha=0.6, label=f'Label {label}', edgecolor='black')
        axes[1].set_title('Text Length (Words) by Label')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f'text_length_by_label_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['length_by_label'] = filepath
        print(f"  ✓ Saved text length by label: {filename}")
    
    return saved_files


def visualize_model_predictions(
    predictions: Union[np.ndarray, pd.Series, List],
    output_dir: Optional[str] = None,
    labels: Optional[Union[np.ndarray, pd.Series, List]] = None,
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, str]:
    """
    Visualize model predictions: distributions, correlations between models.
    
    Args:
        predictions: Model predictions (can be 1D array or 2D array for multiple models)
        output_dir: Output directory for plots
        labels: Optional true labels for comparison
        model_names: Optional names for multiple models
        figsize: Figure size tuple
        
    Returns:
        Dictionary mapping plot type to file path
    """
    if output_dir is None:
        output_dir = _default_visualization_dir()
    _ensure_dir(output_dir)
    timestamp = _get_timestamp()
    saved_files = {}
    
    # Convert to numpy array
    if isinstance(predictions, (pd.Series, list)):
        predictions = np.array(predictions)
    
    # Handle 2D predictions (multiple models)
    if predictions.ndim == 2:
        n_models = predictions.shape[1]
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(n_models)]
    else:
        predictions = predictions.reshape(-1, 1)
        n_models = 1
        model_names = model_names or ['Model']
    
    # 1. Prediction Distributions
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for idx, (pred, name) in enumerate(zip(predictions.T, model_names)):
        ax = axes[idx]
        ax.hist(pred, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f'{name} Prediction Distribution\n(mean={np.mean(pred):.4f}, std={np.std(pred):.4f})')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'model_predictions_distribution_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    saved_files['predictions_distribution'] = filepath
    print(f"  ✓ Saved prediction distributions: {filename}")
    
    # 2. Prediction Correlation (if multiple models)
    if n_models > 1:
        pred_df = pd.DataFrame(predictions, columns=model_names)
        corr_matrix = pred_df.corr()
        
        fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), max(6, n_models * 1.5)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        ax.set_title('Model Predictions Correlation')
        plt.tight_layout()
        filename = f'model_predictions_correlation_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        saved_files['predictions_correlation'] = filepath
        print(f"  ✓ Saved prediction correlation: {filename}")
    
    # 3. Predictions vs True Labels (if labels provided)
    if labels is not None:
        if isinstance(labels, (pd.Series, list)):
            labels = np.array(labels)
        
        # For binary classification, show distribution by label
        if len(set(labels)) == 2:
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
            if n_models == 1:
                axes = [axes]
            
            for idx, (pred, name) in enumerate(zip(predictions.T, model_names)):
                ax = axes[idx]
                for label in sorted(set(labels)):
                    label_pred = pred[labels == label]
                    ax.hist(label_pred, bins=30, alpha=0.6, label=f'Label {label}', edgecolor='black')
                ax.set_title(f'{name} Predictions by True Label')
                ax.set_xlabel('Prediction Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filename = f'model_predictions_vs_labels_{timestamp}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved_files['predictions_vs_labels'] = filepath
            print(f"  ✓ Saved predictions vs labels: {filename}")
    
    return saved_files


def generate_statistics_report(
    df: Optional[pd.DataFrame] = None,
    texts: Optional[Union[List[str], pd.Series]] = None,
    predictions: Optional[Union[np.ndarray, pd.Series, List]] = None,
    output_dir: Optional[str] = None,
    label_col: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate comprehensive statistics report in CSV format.
    
    Args:
        df: DataFrame with tabular features
        texts: Optional text data
        predictions: Optional model predictions
        output_dir: Output directory for CSV files
        label_col: Optional label column name
        
    Returns:
        Dictionary mapping report type to file path
    """
    if output_dir is None:
        output_dir = _default_stats_dir()
    _ensure_dir(output_dir)
    timestamp = _get_timestamp()
    saved_files = {}
    
    # 1. Tabular Features Statistics
    if df is not None:
        stats_list = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove label column if present
        if label_col and label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if label_col and label_col in categorical_cols:
            categorical_cols.remove(label_col)
        
        # Numeric features statistics
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats_list.append({
                    'feature': col,
                    'type': 'numeric',
                    'count': len(data),
                    'missing_count': df[col].isna().sum(),
                    'missing_pct': (df[col].isna().sum() / len(df) * 100),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'unique_count': data.nunique()
                })
        
        # Categorical features statistics
        for col in categorical_cols:
            stats_list.append({
                'feature': col,
                'type': 'categorical',
                'count': df[col].notna().sum(),
                'missing_count': df[col].isna().sum(),
                'missing_pct': (df[col].isna().sum() / len(df) * 100),
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_list)
        filename = f'tabular_statistics_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        stats_df.to_csv(filepath, index=False)
        saved_files['tabular_statistics'] = filepath
        print(f"  ✓ Saved tabular statistics: {filename}")
    
    # 2. Text Features Statistics
    if texts is not None:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        texts = [str(t) if t is not None else '' for t in texts]
        
        char_lengths = [len(text) for text in texts]
        word_lengths = [len(text.split()) for text in texts]
        
        text_stats = {
            'metric': [
                'total_texts',
                'mean_char_length',
                'median_char_length',
                'std_char_length',
                'min_char_length',
                'max_char_length',
                'mean_word_length',
                'median_word_length',
                'std_word_length',
                'min_word_length',
                'max_word_length'
            ],
            'value': [
                len(texts),
                np.mean(char_lengths),
                np.median(char_lengths),
                np.std(char_lengths),
                np.min(char_lengths),
                np.max(char_lengths),
                np.mean(word_lengths),
                np.median(word_lengths),
                np.std(word_lengths),
                np.min(word_lengths),
                np.max(word_lengths)
            ]
        }
        
        text_stats_df = pd.DataFrame(text_stats)
        filename = f'text_statistics_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        text_stats_df.to_csv(filepath, index=False)
        saved_files['text_statistics'] = filepath
        print(f"  ✓ Saved text statistics: {filename}")
    
    # 3. Model Predictions Statistics
    if predictions is not None:
        if isinstance(predictions, (pd.Series, list)):
            predictions = np.array(predictions)
        
        # Handle 2D predictions
        if predictions.ndim == 2:
            n_models = predictions.shape[1]
            model_names = [f'Model_{i+1}' for i in range(n_models)]
        else:
            predictions = predictions.reshape(-1, 1)
            n_models = 1
            model_names = ['Model']
        
        pred_stats_list = []
        for idx, (pred, name) in enumerate(zip(predictions.T, model_names)):
            pred_stats_list.append({
                'model': name,
                'count': len(pred),
                'mean': np.mean(pred),
                'median': np.median(pred),
                'std': np.std(pred),
                'min': np.min(pred),
                'max': np.max(pred),
                'q25': np.percentile(pred, 25),
                'q75': np.percentile(pred, 75)
            })
        
        pred_stats_df = pd.DataFrame(pred_stats_list)
        filename = f'model_predictions_statistics_{timestamp}.csv'
        filepath = os.path.join(output_dir, filename)
        pred_stats_df.to_csv(filepath, index=False)
        saved_files['predictions_statistics'] = filepath
        print(f"  ✓ Saved predictions statistics: {filename}")
    
    return saved_files


def visualize_all_features(
    df: Optional[pd.DataFrame] = None,
    texts: Optional[Union[List[str], pd.Series]] = None,
    predictions: Optional[Union[np.ndarray, pd.Series, List]] = None,
    output_dir: Optional[str] = None,
    label_col: Optional[str] = None,
    labels: Optional[Union[np.ndarray, pd.Series, List]] = None,
    model_names: Optional[List[str]] = None,
    generate_stats: bool = True,
    generate_plots: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Main function to visualize all features and generate statistics.
    
    Args:
        df: DataFrame with tabular features
        texts: Optional text data (list or Series)
        predictions: Optional model predictions
        output_dir: Base output directory
        label_col: Optional label column name in df
        labels: Optional labels for text/prediction analysis
        model_names: Optional names for multiple models
        generate_stats: Whether to generate CSV statistics
        generate_plots: Whether to generate PNG plots
        
    Returns:
        Dictionary with keys 'plots' and 'stats', each containing file paths
    """
    if output_dir is None:
        output_dir = _default_visualization_dir()
    print("=" * 60)
    print("Feature Visualization and Statistics Generation")
    print("=" * 60)
    
    results = {'plots': {}, 'stats': {}}
    
    # Generate plots
    if generate_plots:
        print("\n[1] Generating visualizations...")
        
        if df is not None:
            print("  Visualizing tabular features...")
            tabular_plots = visualize_tabular_features(df, output_dir, label_col)
            results['plots'].update(tabular_plots)
        
        if texts is not None:
            print("  Visualizing text features...")
            text_plots = visualize_text_features(texts, output_dir, labels)
            results['plots'].update(text_plots)
        
        if predictions is not None:
            print("  Visualizing model predictions...")
            pred_plots = visualize_model_predictions(predictions, output_dir, labels, model_names)
            results['plots'].update(pred_plots)
    
    # Generate statistics
    if generate_stats:
        print("\n[2] Generating statistics reports...")
        stats_dir = os.path.join(output_dir, 'stats')
        stats_files = generate_statistics_report(df, texts, predictions, stats_dir, label_col)
        results['stats'] = stats_files
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Generated {len(results['plots'])} plots")
    print(f"  Generated {len(results['stats'])} statistics files")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Feature Visualization Module")
    print("=" * 60)
    print("\nThis is a module, not a standalone script.")
    print("Please use it by importing:")
    print("  from src.vis_stat import plots as viz")
    print("\nOr run the example script:")
    print("  # Example script usage would depend on new location")
    print("=" * 60)


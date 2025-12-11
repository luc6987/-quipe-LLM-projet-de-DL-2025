"""
CamemBERT Visualization Module
CamemBERT 

Provides visualization functions for training curves, embedding quality, feature importance,
confusion matrices, and training reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available. Visualization functions will be limited.")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
    title: str = "Training Curves"
) -> None:
    """
    Plot training curves for loss and accuracy.
    loss 
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Optional validation loss per epoch
        train_accs: Optional training accuracy per epoch
        val_accs: Optional validation accuracy per epoch
        output_path: Path to save the figure
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Cannot plot training curves.")
        return
    
    sns.set_style("whitegrid")
    
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves (if available)
    if train_accs is not None:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
        if val_accs:
            ax.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved training curves to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_quality(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    method: str = 'tsne',
    n_components: int = 2
) -> None:
    """
    Visualize embedding quality using dimensionality reduction.
    
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        labels: Optional labels for coloring
        output_path: Path to save the figure
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_components: Number of components for visualization (2 or 3)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Cannot plot embedding quality.")
        return
    
    try:
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'.")
    except ImportError:
        logger.warning("sklearn not available. Cannot perform dimensionality reduction.")
        return
    
    logger.info(f"Reducing embeddings to {n_components}D using {method.upper()}...")
    embeddings_reduced = reducer.fit_transform(embeddings)
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        ax = fig.add_subplot(111)
        if labels is not None:
            scatter = ax.scatter(
                embeddings_reduced[:, 0],
                embeddings_reduced[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(
                embeddings_reduced[:, 0],
                embeddings_reduced[:, 1],
                alpha=0.6,
                s=20
            )
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            scatter = ax.scatter(
                embeddings_reduced[:, 0],
                embeddings_reduced[:, 1],
                embeddings_reduced[:, 2],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(
                embeddings_reduced[:, 0],
                embeddings_reduced[:, 1],
                embeddings_reduced[:, 2],
                alpha=0.6,
                s=20
            )
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_zlabel(f'{method.upper()} Component 3', fontsize=12)
    
    ax.set_title(f'Embedding Quality Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved embedding quality plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(
    feature_importance: Union[Dict[str, float], pd.Series],
    top_n: int = 20,
    output_path: Optional[Path] = None,
    title: str = "Feature Importance"
) -> None:
    """
    Plot feature importance.
    
    
    Args:
        feature_importance: Dictionary or Series of feature importance scores
        top_n: Number of top features to display
        output_path: Path to save the figure
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Cannot plot feature importance.")
        return
    
    # Convert to Series if needed
    if isinstance(feature_importance, dict):
        importance_series = pd.Series(feature_importance)
    else:
        importance_series = feature_importance
    
    # Sort and get top N
    importance_sorted = importance_series.sort_values(ascending=False).head(top_n)
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(importance_sorted))
    ax.barh(y_pos, importance_sorted.values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_sorted.index, fontsize=10)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved feature importance plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix.
    
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the figure
        title: Plot title
        normalize: Whether to normalize the confusion matrix
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Cannot plot confusion matrix.")
        return
    
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("sklearn not available. Cannot compute confusion matrix.")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        label = 'Normalized'
    else:
        fmt = 'd'
        label = 'Count'
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        ax=ax,
        cbar_kws={'label': label}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_training_report(
    output_dir: Union[str, Path],
    train_scores: Optional[Dict[str, List[float]]] = None,
    val_scores: Optional[Dict[str, List[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save training report with scores and metadata.
    
    
    Args:
        output_dir: Directory to save the report
        train_scores: Dictionary of training scores (e.g., {'AUC': [0.95], 'Accuracy': [0.92]})
        val_scores: Dictionary of validation scores (optional)
        metadata: Additional metadata dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create report dictionary
    report = {
        'timestamp': timestamp,
        'train_scores': train_scores or {},
        'val_scores': val_scores or {},
        'metadata': metadata or {}
    }
    
    # Save as JSON
    report_path = output_dir / f"training_report_{timestamp}.json"
    import json
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Training report saved to {report_path}")
    
    # Create summary text file
    summary_path = output_dir / f"training_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Training Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        if train_scores:
            f.write("Training Scores:\n")
            f.write("-" * 60 + "\n")
            for metric, values in train_scores.items():
                if values:
                    f.write(f"  {metric}: {values[0]:.4f}\n")
            f.write("\n")
        
        if val_scores:
            f.write("Validation Scores:\n")
            f.write("-" * 60 + "\n")
            for metric, values in val_scores.items():
                if values:
                    f.write(f"  {metric}: {values[0]:.4f}\n")
            f.write("\n")
        
        if metadata:
            f.write("Metadata:\n")
            f.write("-" * 60 + "\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
    
    logger.info(f"Training summary saved to {summary_path}")
    
    # Create visualization if scores are available
    if MATPLOTLIB_AVAILABLE and (train_scores or val_scores):
        try:
            plot_path = output_dir / f"training_scores_{timestamp}.png"
            _plot_training_scores(train_scores, val_scores, plot_path)
        except Exception as e:
            logger.warning(f"Could not create training scores plot: {e}")


def _plot_training_scores(
    train_scores: Optional[Dict[str, List[float]]],
    val_scores: Optional[Dict[str, List[float]]],
    output_path: Path
) -> None:
    """Internal function to plot training scores."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    sns.set_style("whitegrid")
    
    # Collect all metrics
    all_metrics = set()
    if train_scores:
        all_metrics.update(train_scores.keys())
    if val_scores:
        all_metrics.update(val_scores.keys())
    
    if not all_metrics:
        return
    
    n_metrics = len(all_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(sorted(all_metrics)):
        ax = axes[idx]
        
        train_val = train_scores.get(metric, [None])[0] if train_scores else None
        val_val = val_scores.get(metric, [None])[0] if val_scores else None
        
        bars = []
        labels = []
        colors = []
        
        if train_val is not None:
            bars.append(train_val)
            labels.append('Train')
            colors.append('steelblue')
        
        if val_val is not None:
            bars.append(val_val)
            labels.append('Validation')
            colors.append('coral')
        
        if bars:
            ax.bar(labels, bars, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, label) in enumerate(zip(bars, labels)):
                ax.text(i, bar, f'{bar:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Training Scores Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


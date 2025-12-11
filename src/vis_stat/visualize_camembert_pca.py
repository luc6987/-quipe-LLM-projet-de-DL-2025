#!/usr/bin/env python3
"""
Visualize CamemBERT PCA Eigenvalues
 CamemBERT PCA 

This script extracts and visualizes all PCA eigenvalues (explained variance) 
from the CamemBERT embedding PCA model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src._imports import np, pd, plt, logging
from src.camembert.embedding import (
    load_camembert_model,
    extract_camembert_embeddings,
    apply_pca_reduction
)
from src.preprocessing.camembert_preprocessing import extract_and_fuse_text_fields
from src.utils.config import load_config
from src.utils.paths import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data_sample(n_samples: int = 10000):
    """Load a sample of training data for PCA fitting."""
    train_file = DATA_DIR / 'train.jsonl'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    logger.info(f"Loading {n_samples} samples from {train_file}...")
    
    import json
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} samples")
    
    return df


def extract_pca_eigenvalues(
    train_file: str = None,
    n_samples: int = 10000,
    n_components: int = 32,
    model_name: str = "camembert-base",
    max_length: int = 256,
    batch_size: int = 32
):
    """
    Extract CamemBERT embeddings, fit PCA, and return eigenvalues.
    
    Args:
        train_file: Path to training JSONL file (optional, uses default if None)
        n_samples: Number of samples to use for PCA fitting
        n_components: Number of PCA components
        model_name: CamemBERT model name
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (PCA model, eigenvalues, explained_variance_ratio)
    """
    # Load config
    try:
        config = load_config()
        semantic_config = config.get('semantic', {})
        n_components = semantic_config.get('pca_components', n_components)
        model_name = semantic_config.get('camembert_model_name', model_name)
        max_length = semantic_config.get('camembert_max_length', max_length)
        batch_size = semantic_config.get('camembert_batch_size', batch_size)
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
    
    # Load training data
    if train_file:
        import json
        logger.info(f"Loading data from {train_file}...")
        data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        df = load_training_data_sample(n_samples)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract and fuse text fields
    logger.info("Extracting and fusing text fields...")
    fused_texts = extract_and_fuse_text_fields(df)
    logger.info(f"Fused text fields: {len(fused_texts)} texts, avg length: {fused_texts.str.len().mean():.1f}")
    
    # Load CamemBERT model
    logger.info(f"Loading CamemBERT model: {model_name}...")
    tokenizer, model, device = load_camembert_model(model_name)
    logger.info(f"Model loaded on device: {device}")
    
    # Extract embeddings
    logger.info("Extracting CamemBERT embeddings (second-to-last layer)...")
    embeddings = extract_camembert_embeddings(
        fused_texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate ALL eigenvalues from original embedding space (before PCA reduction)
    # This gives us all 768 dimensions' eigenvalues
    logger.info("Calculating all eigenvalues from original embedding space (768 dimensions)...")
    
    # Center the data (subtract mean)
    embeddings_centered = embeddings - embeddings.mean(axis=0)
    
    # Calculate covariance matrix
    logger.info("Computing covariance matrix...")
    cov_matrix = np.cov(embeddings_centered.T)
    logger.info(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Compute all eigenvalues and eigenvectors
    logger.info("Computing eigenvalues and eigenvectors...")
    eigenvalues_all, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues_all.argsort()[::-1]
    eigenvalues_all = eigenvalues_all[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate explained variance ratio
    total_variance = eigenvalues_all.sum()
    explained_variance_ratio_all = eigenvalues_all / total_variance
    
    logger.info(f"All eigenvalues shape: {eigenvalues_all.shape}")
    logger.info(f"Total variance: {total_variance:.6f}")
    logger.info(f"Max eigenvalue: {eigenvalues_all.max():.6f}")
    logger.info(f"Min eigenvalue: {eigenvalues_all.min():.6f}")
    logger.info(f"Number of non-zero eigenvalues: {(eigenvalues_all > 1e-10).sum()}")
    
    # Also fit PCA with all components for comparison
    logger.info(f"Fitting PCA with all {embeddings.shape[1]} components for verification...")
    from src._imports import sklearn_PCA as PCA
    pca_all = PCA(n_components=embeddings.shape[1], random_state=42)
    pca_all.fit(embeddings)
    
    # Verify eigenvalues match (should be very close)
    eigenvalues_pca = pca_all.explained_variance_
    logger.info(f"PCA eigenvalues shape: {eigenvalues_pca.shape}")
    logger.info(f"Eigenvalue difference (max): {np.abs(eigenvalues_all - eigenvalues_pca).max():.2e}")
    
    return pca_all, eigenvalues_all, explained_variance_ratio_all, embeddings


def visualize_pca_eigenvalues(
    eigenvalues: np.ndarray,
    explained_variance_ratio: np.ndarray,
    output_path: str = "output/camembert/pca_eigenvalues.png",
    n_components_to_show: int = None
):
    """
    Visualize ALL PCA eigenvalues from original embedding space (before reduction).
    
    Args:
        eigenvalues: Array of ALL eigenvalues (explained variance) from original 768D space
        explained_variance_ratio: Array of explained variance ratios
        output_path: Path to save the visualization
        n_components_to_show: Number of components to show in detail (default: all, or top 100 if > 100)
    """
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_dims = len(eigenvalues)
    if n_components_to_show is None:
        n_components_to_show = min(100, total_dims)  # Show top 100 or all if less than 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'CamemBERT PCA Eigenvalues Analysis (All {total_dims} Dimensions)', 
                 fontsize=16, fontweight='bold')
    
    component_indices = np.arange(1, total_dims + 1)
    component_indices_show = np.arange(1, n_components_to_show + 1)
    
    # 1. Eigenvalues (absolute values) - Bar plot (top N components)
    ax1 = axes[0, 0]
    bars = ax1.bar(component_indices_show, eigenvalues[:n_components_to_show], 
                   alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Dimension Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (Explained Variance)', fontsize=12)
    ax1.set_title(f'Top {n_components_to_show} Eigenvalues (Absolute Values)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(component_indices_show[::max(1, n_components_to_show//20)])
    ax1.set_xticklabels(component_indices_show[::max(1, n_components_to_show//20)], rotation=45, ha='right')
    
    # Add value labels on top bars
    for i, (bar, val) in enumerate(zip(bars[:min(20, n_components_to_show)], eigenvalues[:min(20, n_components_to_show)])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=7)
    
    # 2. Explained Variance Ratio - Bar plot (top N)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(component_indices_show, explained_variance_ratio[:n_components_to_show] * 100, 
                    alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Dimension Index', fontsize=12)
    ax2.set_ylabel('Explained Variance Ratio (%)', fontsize=12)
    ax2.set_title(f'Top {n_components_to_show} Explained Variance Ratio (%)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(component_indices_show[::max(1, n_components_to_show//20)])
    ax2.set_xticklabels(component_indices_show[::max(1, n_components_to_show//20)], rotation=45, ha='right')
    
    # Add cumulative line for all dimensions
    cumulative = np.cumsum(explained_variance_ratio) * 100
    ax2_twin = ax2.twinx()
    ax2_twin.plot(component_indices_show, cumulative[:n_components_to_show], 'r-', linewidth=2, 
                  marker='o', markersize=3, label='Cumulative')
    ax2_twin.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.legend(loc='upper left')
    ax2_twin.grid(True, alpha=0.3, linestyle='--')
    
    # 3. ALL Eigenvalues - Line plot (log scale, all dimensions)
    ax3 = axes[1, 0]
    # Plot all dimensions, but use different markers for visibility
    ax3.plot(component_indices, eigenvalues, '-', linewidth=1.5, color='steelblue', alpha=0.7, label='All eigenvalues')
    # Highlight top 32 components
    ax3.plot(component_indices[:32], eigenvalues[:32], 'o', markersize=4, color='red', 
             label='Top 32 (used in PCA)', alpha=0.8)
    ax3.set_xlabel('Dimension Index (All 768 Dimensions)', fontsize=12)
    ax3.set_ylabel('Eigenvalue (Log Scale)', fontsize=12)
    ax3.set_title(f'All {total_dims} Eigenvalues (Log Scale)', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend()
    # Set x-axis to show every 100th dimension
    x_ticks = np.arange(0, total_dims, max(50, total_dims//20))
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_ticks, rotation=45, ha='right')
    
    # 4. Scree plot (All eigenvalues, linear and log scale overlay)
    ax4 = axes[1, 1]
    # Linear scale for first part
    ax4.plot(component_indices[:100], eigenvalues[:100], 'o-', linewidth=2, markersize=3, 
             color='darkgreen', label='First 100 (linear)', alpha=0.7)
    ax4.fill_between(component_indices[:100], 0, eigenvalues[:100], alpha=0.2, color='lightgreen')
    ax4.set_xlabel('Dimension Index', fontsize=12)
    ax4.set_ylabel('Eigenvalue', fontsize=12)
    ax4.set_title('Scree Plot (First 100 Dimensions)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(component_indices[:100:10])
    ax4.set_xticklabels(component_indices[:100:10], rotation=45, ha='right')
    ax4.legend()
    
    # Add statistics text
    non_zero = (eigenvalues > 1e-10).sum()
    stats_text = (
        f"Total Dimensions: {len(eigenvalues)}\n"
        f"Non-zero Eigenvalues: {non_zero}\n"
        f"Total Explained Variance: {explained_variance_ratio.sum()*100:.2f}%\n"
        f"Max Eigenvalue: {eigenvalues.max():.6f}\n"
        f"Min Eigenvalue: {eigenvalues.min():.6f}\n"
        f"Mean Eigenvalue: {eigenvalues.mean():.6f}\n"
        f"Median Eigenvalue: {np.median(eigenvalues):.6f}\n"
        f"Top 32 Explained: {explained_variance_ratio[:32].sum()*100:.2f}%"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualization saved to: {output_path}")
    
    # Also save eigenvalues to CSV
    csv_path = output_path.replace('.png', '.csv')
    df_eigenvalues = pd.DataFrame({
        'component_index': component_indices,
        'eigenvalue': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_explained_variance': cumulative / 100
    })
    df_eigenvalues.to_csv(csv_path, index=False)
    logger.info(f"✓ Eigenvalues data saved to: {csv_path}")
    
    plt.close()
    
    return df_eigenvalues


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize CamemBERT PCA eigenvalues')
    parser.add_argument('--train-file', type=str, default=None,
                       help='Path to training JSONL file (uses default if not provided)')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of samples to use for PCA fitting (default: 10000)')
    parser.add_argument('--n-components', type=int, default=32,
                       help='Number of PCA components (default: 32, overridden by config)')
    parser.add_argument('--output', type=str, default='output/camembert/pca_eigenvalues.png',
                       help='Output path for visualization (default: output/camembert/pca_eigenvalues.png)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("CamemBERT PCA Eigenvalues Visualization")
    logger.info("="*60)
    
    try:
        # Extract PCA eigenvalues
        pca, eigenvalues, explained_variance_ratio, reduced_embeddings = extract_pca_eigenvalues(
            train_file=args.train_file,
            n_samples=args.n_samples,
            n_components=args.n_components
        )
        
        # Visualize
        df_eigenvalues = visualize_pca_eigenvalues(
            eigenvalues,
            explained_variance_ratio,
            output_path=args.output,
            n_components_to_show=100  # Show top 100 in detail
        )
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PCA Eigenvalues Summary (All Dimensions)")
        logger.info("="*60)
        logger.info(f"Total dimensions: {len(eigenvalues)}")
        logger.info(f"Non-zero eigenvalues: {(eigenvalues > 1e-10).sum()}")
        logger.info(f"Total explained variance: {explained_variance_ratio.sum()*100:.2f}%")
        logger.info(f"Top 32 explained variance: {explained_variance_ratio[:32].sum()*100:.2f}%")
        logger.info(f"Max eigenvalue: {eigenvalues.max():.6f}")
        logger.info(f"Min eigenvalue: {eigenvalues.min():.6f}")
        logger.info(f"Mean eigenvalue: {eigenvalues.mean():.6f}")
        logger.info(f"Median eigenvalue: {np.median(eigenvalues):.6f}")
        logger.info("\nTop 20 Dimensions by Eigenvalue:")
        top_20 = df_eigenvalues.head(20)
        for _, row in top_20.iterrows():
            logger.info(f"  Dim{int(row['component_index']):3d}: eigenvalue={row['eigenvalue']:.6f}, "
                       f"explained_var={row['explained_variance_ratio']*100:.4f}%, "
                       f"cumulative={row['cumulative_explained_variance']*100:.2f}%")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


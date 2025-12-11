"""
Example script demonstrating how to use the visualization module.

"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.vis_stat import plots as viz
from src.preprocessing import data_processing as dp
from src.utils.paths import DATA_DIR, get_visualizations_dir


def example_visualize_tabular_features():
    """Example: Visualize tabular features from training data."""
    print("Example 1: Visualizing tabular features")
    print("-" * 60)
    
    # Load data
    train_path = str(DATA_DIR / 'train.jsonl')
    
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Skipping example.")
        return
    
    # Load and engineer features
    df = dp.load_data(train_path, clean_text=True)
    df_engineered, transformers = dp.engineer_features(df, is_train=True)
    
    # Visualize
    output_dir = str(get_visualizations_dir())
    results = viz.visualize_tabular_features(
        df_engineered,
        output_dir=output_dir,
        label_col='label'
    )
    
    print(f"\nGenerated {len(results)} plots:")
    for plot_type, filepath in results.items():
        print(f"  - {plot_type}: {filepath}")


def example_visualize_text_features():
    """Example: Visualize text features."""
    print("\nExample 2: Visualizing text features")
    print("-" * 60)
    
    # Load data
    train_path = str(DATA_DIR / 'train.jsonl')
    
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Skipping example.")
        return
    
    # Load data
    df = dp.load_data(train_path, clean_text=True)
    
    # Prepare text column
    if 'text' not in df.columns:
        if 'extended_tweet.full_text' in df.columns:
            df['text'] = df['extended_tweet.full_text'].fillna('')
        else:
            print("Warning: No text column found. Skipping example.")
            return
    
    texts = df['text'].fillna('').astype(str).tolist()
    labels = df['label'].values if 'label' in df.columns else None
    
    # Visualize
    output_dir = str(get_visualizations_dir())
    results = viz.visualize_text_features(
        texts,
        output_dir=output_dir,
        labels=labels
    )
    
    print(f"\nGenerated {len(results)} plots:")
    for plot_type, filepath in results.items():
        print(f"  - {plot_type}: {filepath}")


def example_generate_statistics():
    """Example: Generate statistics reports."""
    print("\nExample 3: Generating statistics reports")
    print("-" * 60)
    
    # Load data
    train_path = str(DATA_DIR / 'train.jsonl')
    
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Skipping example.")
        return
    
    # Load and engineer features
    df = dp.load_data(train_path, clean_text=True)
    df_engineered, transformers = dp.engineer_features(df, is_train=True)
    
    # Prepare text
    if 'text' in df.columns:
        texts = df['text'].fillna('').astype(str).tolist()
    else:
        texts = None
    
    # Generate statistics
    stats_dir = str(get_visualizations_dir() / 'stats')
    results = viz.generate_statistics_report(
        df=df_engineered,
        texts=texts,
        output_dir=stats_dir,
        label_col='label'
    )
    
    print(f"\nGenerated {len(results)} statistics files:")
    for stat_type, filepath in results.items():
        print(f"  - {stat_type}: {filepath}")


def example_visualize_all():
    """Example: Use the main function to visualize everything."""
    print("\nExample 4: Visualizing all features using main function")
    print("-" * 60)
    
    # Load data
    train_path = str(DATA_DIR / 'train.jsonl')
    
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Skipping example.")
        return
    
    # Load and engineer features
    df = dp.load_data(train_path, clean_text=True)
    df_engineered, transformers = dp.engineer_features(df, is_train=True)
    
    # Prepare text
    if 'text' in df.columns:
        texts = df['text'].fillna('').astype(str)
    else:
        texts = None
    
    # Prepare labels
    labels = df['label'].values if 'label' in df.columns else None
    
    # Visualize everything
    output_dir = str(get_visualizations_dir())
    results = viz.visualize_all_features(
        df=df_engineered,
        texts=texts,
        output_dir=output_dir,
        label_col='label',
        labels=labels,
        generate_stats=True,
        generate_plots=True
    )
    
    print(f"\nSummary:")
    print(f"  Plots generated: {len(results['plots'])}")
    print(f"  Statistics files generated: {len(results['stats'])}")


if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Module Examples")
    print("=" * 60)
    
    # Run examples
    example_visualize_tabular_features()
    example_visualize_text_features()
    example_generate_statistics()
    example_visualize_all()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


#!/usr/bin/python3
"""
DeBERTa Embeddings Validation Script
DeBERTa 嵌入向量验证脚本

This script loads and validates the extracted embeddings from the fine-tuned DeBERTa model.
It checks the format, dimensions, and can optionally display sample embeddings.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add project root to path to allow imports from src
# This ensures the script can be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# From src/utils/bertweet_lora/ -> go up 3 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup HuggingFace cache using unified path configuration
# MUST be set BEFORE importing transformers/huggingface_hub
from src.utils.paths import setup_huggingface_cache, get_embeddings_dir, get_models_output_dir
setup_huggingface_cache()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_DIR = str(get_embeddings_dir())
MODEL_OUTPUT_DIR = str(get_models_output_dir() / "bertweet_lora")

# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------
def load_embeddings(embedding_path: str):
    """Load embeddings from pickle file."""
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    with open(embedding_path, "rb") as f:
        data = pickle.load(f)
    
    return data


def validate_embeddings(data: dict, split_name: str):
    """Validate embedding format and dimensions."""
    print(f"\n{'='*60}")
    print(f"Validating {split_name} embeddings")
    print(f"{'='*60}")
    
    # Check required keys
    if "embeddings" not in data:
        raise ValueError(f"{split_name}: Missing 'embeddings' key")
    
    embeddings = data["embeddings"]
    
    # Validate embeddings
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"{split_name}: Embeddings must be numpy array, got {type(embeddings)}")
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  - Number of samples: {embeddings.shape[0]}")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")
    
    # Check for NaN or Inf
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    if nan_count > 0:
        print(f"  ⚠ Warning: Found {nan_count} NaN values")
    if inf_count > 0:
        print(f"  ⚠ Warning: Found {inf_count} Inf values")
    if nan_count == 0 and inf_count == 0:
        print(f"  ✓ No NaN or Inf values")
    
    # Check statistics
    print(f"  - Mean: {embeddings.mean():.6f}")
    print(f"  - Std: {embeddings.std():.6f}")
    print(f"  - Min: {embeddings.min():.6f}")
    print(f"  - Max: {embeddings.max():.6f}")
    
    # Check labels if available (for train/val splits)
    if "labels" in data:
        labels = data["labels"]
        if not isinstance(labels, np.ndarray):
            raise TypeError(f"{split_name}: Labels must be numpy array, got {type(labels)}")
        print(f"✓ Labels shape: {labels.shape}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  - Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"    Label {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    else:
        print(f"  ℹ No labels (expected for test set)")
    
    # Check indices if available
    if "indices" in data:
        indices = data["indices"]
        print(f"✓ Indices shape: {indices.shape}")
        print(f"  - Index range: [{indices.min()}, {indices.max()}]")
    
    return True


def main():
    """Main validation function."""
    print("="*60)
    print("DeBERTa Embeddings Validation")
    print("="*60)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_dir = os.path.join(script_dir, EMBEDDING_DIR)
    
    if not os.path.exists(embedding_dir):
        print(f"Error: Embedding directory not found: {embedding_dir}")
        print("Please run the training script first to generate embeddings.")
        sys.exit(1)
    
    # Find all embedding files
    embedding_files = {}
    for filename in os.listdir(embedding_dir):
        if filename.endswith("_embeddings.pkl"):
            split_name = filename.replace("_embeddings.pkl", "")
            embedding_files[split_name] = os.path.join(embedding_dir, filename)
    
    if not embedding_files:
        print(f"Error: No embedding files found in {embedding_dir}")
        print("Expected files: train_embeddings.pkl, val_embeddings.pkl, test_embeddings.pkl")
        sys.exit(1)
    
    print(f"\nFound {len(embedding_files)} embedding file(s):")
    for split_name, filepath in embedding_files.items():
        print(f"  - {split_name}: {filepath}")
    
    # Validate each embedding file
    all_valid = True
    for split_name, filepath in embedding_files.items():
        try:
            data = load_embeddings(filepath)
            validate_embeddings(data, split_name)
        except Exception as e:
            print(f"\n✗ Error validating {split_name}: {e}")
            all_valid = False
    
    # Summary
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ All embeddings validated successfully!")
    else:
        print("✗ Some embeddings failed validation")
    print(f"{'='*60}")
    
    # Optional: Display sample embeddings
    if len(embedding_files) > 0:
        first_split = list(embedding_files.keys())[0]
        first_file = embedding_files[first_split]
        data = load_embeddings(first_file)
        print(f"\nSample embedding (first 10 dimensions from {first_split}):")
        print(data["embeddings"][0, :10])


if __name__ == "__main__":
    main()


"""
Library Builder Module
 (Hashtags/Mentions)

Scans raw data to build frequency-based dictionaries for hashtags and user mentions.
 Hashtag  @ 

Outputs:
- hashtag_library.json
- mention_library.json
- Statistical plots
"""

import json
import logging
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def build_libraries(
    texts: List[str], 
    top_k_hashtags: int = 1000, 
    top_k_mentions: int = 1000,
    output_dir: Union[str, Path] = "output/preprocessing/visualizations"
) -> Dict[str, Dict]:
    """
    Build hashtag and mention libraries from list of texts.
    
    Args:
        texts: List of raw text strings
        top_k_hashtags: Number of top hashtags to keep
        top_k_mentions: Number of top mentions to keep
        output_dir: Directory to save JSON libraries and plots
        
    Returns:
        Dictionary containing the libraries path or content
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Regex patterns
    hashtag_pattern = re.compile(r"#(\w+)")
    mention_pattern = re.compile(r"@(\w+)")
    
    hashtag_counter = Counter()
    mention_counter = Counter()
    
    logger.info(f"Scanning {len(texts)} texts for libraries...")
    
    for text in texts:
        if not isinstance(text, str):
            continue
            
        # Lowercase for consistency
        text_lower = text.lower()
        
        hashtags = hashtag_pattern.findall(text_lower)
        mentions = mention_pattern.findall(text_lower)
        
        hashtag_counter.update(hashtags)
        mention_counter.update(mentions)
        
    logger.info(f"Found {len(hashtag_counter)} unique hashtags, {len(mention_counter)} unique mentions")
    
    # Select Top K
    top_hashtags = dict(hashtag_counter.most_common(top_k_hashtags))
    top_mentions = dict(mention_counter.most_common(top_k_mentions))
    
    # Save JSON
    lib_dir = output_dir / "libraries"
    lib_dir.mkdir(parents=True, exist_ok=True)
    
    hashtag_lib_path = lib_dir / "hashtag_library.json"
    mention_lib_path = lib_dir / "mention_library.json"
    
    with open(hashtag_lib_path, 'w', encoding='utf-8') as f:
        json.dump(top_hashtags, f, indent=2)
        
    with open(mention_lib_path, 'w', encoding='utf-8') as f:
        json.dump(top_mentions, f, indent=2)
        
    logger.info(f"Saved libraries to {lib_dir}")
    
    # Visualization
    plot_dir = output_dir
    _plot_top_counts(top_hashtags, "Top Hashtags", plot_dir / "top_hashtags.png")
    _plot_top_counts(top_mentions, "Top Mentions", plot_dir / "top_mentions.png")
    
    return {
        "hashtag_lib": top_hashtags,
        "mention_lib": top_mentions,
        "paths": {
            "hashtag_lib_path": str(hashtag_lib_path),
            "mention_lib_path": str(mention_lib_path)
        }
    }

def _plot_top_counts(counts: Dict[str, int], title: str, save_path: Path):
    """Helper to plot top counts."""
    if not counts:
        return
        
    # Take top 20 for plot
    top_20 = dict(list(counts.items())[:20])
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(top_20.values()), y=list(top_20.keys()), palette="viridis")
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


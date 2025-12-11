"""
Error Analysis Module
错误分析模块

Handles logging of misclassified samples (False Positives / False Negatives) during validation.
处理验证期间错误分类样本（假阳性/假阴性）的记录。
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

def log_error_statistics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series], 
    ids: Union[np.ndarray, pd.Series],
    output_dir: Union[str, Path] = "output/monitoring",
    filename_prefix: str = "error"
) -> Dict:
    """
    Identify and log error entries (FP/FN) to JSON and CSV.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (0/1)
        ids: Sample IDs corresponding to y_true/y_pred
        output_dir: Output directory
        filename_prefix: Prefix for output files
        
    Returns:
        Dictionary containing stats summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure numpy format
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ids = np.array(ids)
    
    # Identify Indices
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    
    fp_ids = ids[fp_indices].tolist()
    fn_ids = ids[fn_indices].tolist()
    
    # Statistics
    stats = {
        "total_samples": len(y_true),
        "false_positives_count": len(fp_ids),
        "false_negatives_count": len(fn_ids),
        "accuracy": (y_true == y_pred).mean(),
        "fp_ratio": len(fp_ids) / len(y_true),
        "fn_ratio": len(fn_ids) / len(y_true),
        "false_positives_ids": fp_ids,
        "false_negatives_ids": fn_ids
    }
    
    # Save JSON details (with IDs)
    json_path = output_dir / f"{filename_prefix}_statistics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
        
    # Save CSV Summary (without ID list for lighter reading)
    summary_stats = {k: v for k, v in stats.items() if not isinstance(v, list)}
    summary_df = pd.DataFrame([summary_stats])
    csv_path = output_dir / f"{filename_prefix}_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    logger.info(f"Error statistics saved to {json_path}")
    logger.info(f"  FP: {len(fp_ids)}, FN: {len(fn_ids)}")
    
    return stats

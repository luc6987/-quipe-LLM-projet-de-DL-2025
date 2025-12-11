"""
Training Monitor Module
训练监控模块

Functions to visualize training progress (metrics over epochs) and model performance (Confusion Matrix, ROC, etc.).
用于可视化训练进度（各Epoch的指标）和模型性能（混淆矩阵、ROC等）的函数。
"""

import matplotlib
matplotlib.use('Agg') # Non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import logging

logger = logging.getLogger(__name__)

def plot_training_history(evals_result: Dict, output_dir: Union[str, Path], test_metric_values: Optional[Dict[str, List[float]]] = None):
    """
    Plot training curves for all metrics in evals_result.
    绘制 evals_result 中所有指标的训练曲线。
    
    Args:
        evals_result: Result dictionary from xgb.train / model.fit
        output_dir: Directory to save plots.
        test_metric_values: Optional dict of test set metric values, e.g., {'binary_logloss': [0.5, 0.4, ...]}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Structure of evals_result: {'validation_0': {'logloss': [...], 'error': [...]}, 'validation_1': ...}
    # Usually: validation_0 = Train, validation_1 = Val (depending on eval_set order)
    
    # Identify datasets (keys like 'validation_0', 'validation_1')
    datasets = list(evals_result.keys())
    if not datasets:
        logger.warning("No datasets found in evals_result to plot.")
        return
        
    # Inspect first dataset to find metrics
    metrics = list(evals_result[datasets[0]].keys())
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for i, dataset in enumerate(datasets):
            label_name = "Train" if i == 0 else "Validation" # Heuristic naming
            # Or use original key if preferred
            # label_name = dataset 
            
            values = evals_result[dataset][metric]
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=f'{label_name}')
        
        # Add test set curve if provided
        # Handle metric name mapping (e.g., 'logloss' vs 'binary_logloss')
        if test_metric_values:
            # Try exact match first
            test_metric_key = metric
            if metric not in test_metric_values:
                # Try alternative names
                if metric == 'logloss' and 'binary_logloss' in test_metric_values:
                    test_metric_key = 'binary_logloss'
                elif metric == 'binary_logloss' and 'logloss' in test_metric_values:
                    test_metric_key = 'logloss'
            
            if test_metric_key in test_metric_values:
                test_values = test_metric_values[test_metric_key]
                test_epochs = range(1, len(test_values) + 1)
                plt.plot(test_epochs, test_values, label='Test', linestyle='--', linewidth=2, color='green')
            
        plt.title(f'Training Curve - {metric.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = output_dir / f"training_curve_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved training curve: {save_path}")

def plot_confusion_matrix(y_true, y_pred, output_dir: Union[str, Path]):
    """
    Plot Confusion Matrix.
    绘制混淆矩阵。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Obs (0)', 'Inf (1)'],
                yticklabels=['Obs (0)', 'Inf (1)'])
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved confusion matrix: {save_path}")

def plot_roc_curve(y_true, y_pred_proba, output_dir: Union[str, Path]):
    """
    Plot ROC Curve and calculate AUC.
    绘制 ROC 曲线并计算 AUC。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = output_dir / "roc_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved ROC curve: {save_path}")

def plot_classification_report(y_true, y_pred, output_dir: Union[str, Path]):
    """
    Visualize Classification Report (Precision, Recall, F1) as Heatmap.
    将分类报告可视化为热图。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    # Convert to DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    # Drop support to visualize scores only
    df_scores = df_report.drop(columns=['support'])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_scores, annot=True, cmap='RdYlGn', fmt='.3f', vmin=0.0, vmax=1.0)
    plt.title('Classification Report')
    
    save_path = output_dir / "classification_report.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved classification report: {save_path}")

def generate_training_report(
    evals_result: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: Union[str, Path],
    test_metric_values: Optional[Dict[str, List[float]]] = None
):
    """
    Wrapper to generate all training monitoring plots.
    
    Args:
        evals_result: Evaluation results from model training
        y_true: True labels for validation set
        y_pred: Predicted labels for validation set
        y_pred_proba: Predicted probabilities for validation set
        output_dir: Directory to save plots
        test_metric_values: Optional dict of test set metric values per epoch
    """
    plot_training_history(evals_result, output_dir, test_metric_values=test_metric_values)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_roc_curve(y_true, y_pred_proba, output_dir)
    plot_classification_report(y_true, y_pred, output_dir)

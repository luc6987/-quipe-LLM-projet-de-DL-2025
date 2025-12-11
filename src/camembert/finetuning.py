"""
CamemBERT Fine-tuning Module
CamemBERT 微调模块

Handles fine-tuning of CamemBERT with classification head, including training visualization
and overfitting/underfitting detection.
"""

# 从统一导入模块导入所有库
from .._imports import (
    os, json, np, pd, Path, Dict, Tuple, Optional, List, datetime, logging,
    TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, torch,
    CamembertTokenizer, CamembertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    torch_Dataset as Dataset, EVALUATE_AVAILABLE, evaluate
)
from typing import TYPE_CHECKING

# Import PREFIX_CHECKPOINT_DIR from transformers
try:
    if TRANSFORMERS_AVAILABLE:
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    else:
        PREFIX_CHECKPOINT_DIR = None
except ImportError:
    PREFIX_CHECKPOINT_DIR = None

logger = logging.getLogger(__name__)

from ..preprocessing.camembert_preprocessing import (
    extract_and_fuse_text_fields,
    classify_and_replace_urls,
    normalize_text_for_camembert
)


class FineTuningCallback:
    """Callback to collect training metrics for visualization."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []
    
    def on_epoch_end(self, epoch, train_loss, val_loss=None, train_acc=None, val_acc=None):
        """Record metrics at end of epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accs.append(train_acc)
        if val_acc is not None:
            self.val_accs.append(val_acc)


def create_finetuning_model(model_name: str = "camembert-base", num_labels: int = 2):
    """
    Create CamemBERT model with classification head.
    创建带分类头的 CamemBERT 模型。
    
    Args:
        model_name: Pre-trained model name
        num_labels: Number of classification labels (default: 2)
        
    Returns:
        Tuple of (tokenizer, model)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for fine-tuning")
    
    logger.info(f"Creating fine-tuning model: {model_name}")
    
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    logger.info(f"Model created with {num_labels} labels")
    return tokenizer, model


def prepare_finetuning_data(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 256,
    text_column: Optional[str] = None,
    preprocessed_texts: Optional[pd.Series] = None
) -> Tuple['torch.utils.data.Dataset', 'torch.utils.data.Dataset']:
    """
    Prepare data for fine-tuning.
    准备微调数据。
    
    Args:
        df: DataFrame with text and label columns
        tokenizer: CamemBERT tokenizer
        max_length: Maximum sequence length
        text_column: Name of text column (if None, will fuse text fields)
        preprocessed_texts: Optional preprocessed text Series (already fused, URL-replaced, normalized)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Preparing fine-tuning data...")
    
    # Use preprocessed texts if provided (already processed with same pipeline as inference)
    if preprocessed_texts is not None:
        logger.info("Using preprocessed texts (already fused, URL-replaced, normalized)...")
        normalized_texts = preprocessed_texts
    else:
        # Extract and fuse text fields
        if text_column is None or text_column not in df.columns:
            logger.info("Fusing text fields...")
            fused_texts = extract_and_fuse_text_fields(df)
        else:
            fused_texts = df[text_column]
        
        # Classify and replace URLs
        logger.info("Classifying and replacing URLs...")
        fused_texts, _ = classify_and_replace_urls(df, fused_texts)
        
        # Normalize text
        logger.info("Normalizing text...")
        normalized_texts = normalize_text_for_camembert(fused_texts)
    
    # Get labels
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'label' column")
    labels = df['label'].values
    
    # Tokenize
    logger.info("Tokenizing texts...")
    encodings = tokenizer(
        normalized_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Check if torch is available
    if not TORCH_AVAILABLE or torch is None:
        raise ImportError("PyTorch is required for fine-tuning data preparation")
    
    # Create dataset
    class TweetDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
    
    dataset = TweetDataset(encodings, labels)
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset


def detect_overfitting(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Detect overfitting and underfitting patterns.
    检测过拟合和欠拟合模式。
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Optional training accuracy per epoch
        val_accs: Optional validation accuracy per epoch
        
    Returns:
        Dictionary with detection results
    """
    results = {
        'overfitting_detected': False,
        'underfitting_detected': False,
        'overfitting_epoch': None,
        'best_epoch': None,
        'best_val_loss': None,
        'analysis': []
    }
    
    if len(val_losses) < 2:
        return results
    
    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]
    results['best_epoch'] = int(best_epoch)
    results['best_val_loss'] = float(best_val_loss)
    
    # Check for overfitting: val_loss increases while train_loss decreases
    for i in range(1, len(val_losses)):
        if val_losses[i] > val_losses[i-1] and train_losses[i] < train_losses[i-1]:
            results['overfitting_detected'] = True
            results['overfitting_epoch'] = int(i)
            results['analysis'].append(
                f"Overfitting detected at epoch {i}: "
                f"val_loss increased ({val_losses[i-1]:.4f} -> {val_losses[i]:.4f}) "
                f"while train_loss decreased ({train_losses[i-1]:.4f} -> {train_losses[i]:.4f})"
            )
            break
    
    # Check for underfitting: both losses are high and not decreasing
    if len(train_losses) >= 3:
        recent_train_trend = np.mean(np.diff(train_losses[-3:]))
        recent_val_trend = np.mean(np.diff(val_losses[-3:]))
        
        if recent_train_trend > -0.01 and recent_val_trend > -0.01:
            if train_losses[-1] > 0.5 and val_losses[-1] > 0.5:
                results['underfitting_detected'] = True
                results['analysis'].append(
                    f"Underfitting detected: both losses are high "
                    f"(train: {train_losses[-1]:.4f}, val: {val_losses[-1]:.4f}) "
                    f"and not decreasing significantly"
                )
    
    # Check gap between train and val loss
    if len(train_losses) > 0 and len(val_losses) > 0:
        final_gap = val_losses[-1] - train_losses[-1]
        if final_gap > 0.3:
            results['analysis'].append(
                f"Large gap between train and validation loss: {final_gap:.4f}. "
                f"This may indicate overfitting."
            )
    
    return results


def _save_monitoring_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]],
    val_accs: Optional[List[float]],
    val_labels: np.ndarray,
    val_pred_binary: np.ndarray,
    val_pred_proba: np.ndarray,
    output_dir: Path,
    overfitting_info: Optional[Dict] = None
) -> None:
    """
    Save monitoring plots similar to CatBoost/LightGBM.
    保存监控图表（与CatBoost/LightGBM类似）。
    
    Generates:
    - Training curves (loss, AUC, accuracy)
    - ROC curve
    - Confusion matrix
    - PR curve
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, classification_report
    except ImportError:
        logger.warning("matplotlib/seaborn/sklearn not available. Skipping monitoring plots.")
        return
    
    sns.set_style("whitegrid")
    
    # 1. Training curves - Loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    if val_losses:
        plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    if overfitting_info and overfitting_info.get('best_epoch') is not None:
        best_epoch = overfitting_info['best_epoch']
        best_val_loss = overfitting_info['best_val_loss']
        plt.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch + 1})')
        plt.plot(best_epoch + 1, best_val_loss, 'go', markersize=10)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curve - Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve_logloss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training curves - Accuracy (if available)
    if train_accs and val_accs:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accs) + 1)
        plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
        plt.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        
        if overfitting_info and overfitting_info.get('best_epoch') is not None:
            best_epoch = overfitting_info['best_epoch']
            plt.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.5)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training Curve - Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve_auc.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. ROC curve
    fpr, tpr, _ = roc_curve(val_labels, val_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confusion matrix
    cm = confusion_matrix(val_labels, val_pred_binary)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Obs (0)', 'Inf (1)'],
                yticklabels=['Obs (0)', 'Inf (1)'])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=axes[1],
                xticklabels=['Obs (0)', 'Inf (1)'],
                yticklabels=['Obs (0)', 'Inf (1)'])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. PR curve
    precision, recall, _ = precision_recall_curve(val_labels, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Classification report visualization
    report_dict = classification_report(val_labels, val_pred_binary, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_scores = df_report.drop(columns=['support'], errors='ignore')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_scores, annot=True, cmap='RdYlGn', fmt='.3f', vmin=0.0, vmax=1.0)
    plt.title('Classification Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "classification_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved monitoring plots to {output_dir}")


def plot_finetuning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
    overfitting_info: Optional[Dict] = None
) -> None:
    """
    Plot fine-tuning curves with overfitting detection.
    绘制微调曲线，包含过拟合检测。
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Optional training accuracy per epoch
        val_accs: Optional validation accuracy per epoch
        output_path: Path to save the figure
        overfitting_info: Optional overfitting detection results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Determine the number of epochs based on available data
    max_len = max(len(train_losses) if train_losses else 0, 
                  len(val_losses) if val_losses else 0)
    
    if max_len == 0:
        logger.warning("No training data available for plotting")
        if output_path:
            plt.close()
        return
    
    epochs = range(1, max_len + 1)
    
    # Plot 1: Loss curves
    ax = axes[0]
    if train_losses and len(train_losses) >= max_len:
        ax.plot(epochs, train_losses[:max_len], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    if val_losses and len(val_losses) >= max_len:
        ax.plot(epochs, val_losses[:max_len], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    # Mark best epoch
    if overfitting_info and overfitting_info.get('best_epoch') is not None:
        best_epoch = overfitting_info['best_epoch']
        best_val_loss = overfitting_info['best_val_loss']
        ax.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch + 1})')
        ax.plot(best_epoch + 1, best_val_loss, 'go', markersize=10, label='Best Val Loss')
    
    # Mark overfitting point
    if overfitting_info and overfitting_info.get('overfitting_epoch') is not None:
        overfit_epoch = overfitting_info['overfitting_epoch']
        ax.axvline(x=overfit_epoch + 1, color='orange', linestyle='--', alpha=0.7, label='Overfitting Detected')
        ax.plot(overfit_epoch + 1, val_losses[overfit_epoch], 'o', color='orange', markersize=10)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Fine-tuning Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves (if available)
    if train_accs is not None and val_accs is not None:
        ax = axes[1]
        ax.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
        ax.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
        
        if overfitting_info and overfitting_info.get('best_epoch') is not None:
            best_epoch = overfitting_info['best_epoch']
            ax.axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Fine-tuning Accuracy Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved fine-tuning curves to {output_path}")
    else:
        plt.show()
    
    plt.close()


def train_camembert_classifier(
    df: pd.DataFrame,
    model_name: str = "camembert-base",
    output_dir: Optional[Path] = None,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
    validation_split: float = 0.2,
    early_stopping_patience: int = 2,
    save_plots: bool = True,
    visualization_dir: Optional[Path] = None,
    preprocessed_texts: Optional[pd.Series] = None
) -> Tuple:
    """
    Fine-tune CamemBERT classifier with visualization.
    微调 CamemBERT 分类器，包含可视化。
    
    Args:
        df: DataFrame with text and label columns
        model_name: Pre-trained model name
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        max_length: Maximum sequence length
        validation_split: Validation set proportion
        early_stopping_patience: Early stopping patience
        save_plots: Whether to save visualization plots
        visualization_dir: Directory for visualization output
        
    Returns:
        Tuple of (fine-tuned model, tokenizer, training history)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for fine-tuning")
    
    logger.info("="*60)
    logger.info("CamemBERT Fine-tuning")
    logger.info("="*60)
    
    # Create model
    tokenizer, model = create_finetuning_model(model_name, num_labels=2)
    
    # Prepare data
    # Use preprocessed texts if provided (for unified preprocessing)
    if preprocessed_texts is not None:
        logger.info("Using preprocessed texts (unified preprocessing pipeline)")
    
    train_dataset, val_dataset = prepare_finetuning_data(
        df, tokenizer, max_length=max_length, preprocessed_texts=preprocessed_texts
    )
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("output/models/camembert_finetuned")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup visualization directory
    if visualization_dir is None:
        visualization_dir = Path("output/camembert/finetuning")
    visualization_dir = Path(visualization_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Metrics
    if EVALUATE_AVAILABLE:
        accuracy_metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy_metric.compute(predictions=predictions, references=labels)
    else:
        # Fallback: compute accuracy manually
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": float(accuracy)}
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > 0 else [],
    )
    
    # Train
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    logger.info(f"Fine-tuning complete!")
    logger.info(f"  Final train loss: {train_result.training_loss:.4f}")
    logger.info(f"  Final eval loss: {eval_result['eval_loss']:.4f}")
    logger.info(f"  Final eval accuracy: {eval_result['eval_accuracy']:.4f}")
    
    # Extract training history
    history = trainer.state.log_history
    
    # Collect metrics for visualization (per epoch)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Track metrics per epoch using dictionaries
    epoch_train_losses = {}
    epoch_val_losses = {}
    epoch_val_accs = {}
    
    # Extract epoch-level metrics
    for log in history:
        if 'epoch' in log:
            epoch = int(log['epoch'])
            
            # Collect validation metrics (only at epoch end)
            if 'eval_loss' in log:
                epoch_val_losses[epoch] = log['eval_loss']
            if 'eval_accuracy' in log:
                epoch_val_accs[epoch] = log['eval_accuracy']
            
            # For training loss, use the last training loss value for each epoch
            # (since training loss is logged per step, we take the last one per epoch)
            if 'loss' in log and 'eval_loss' not in log and 'train_runtime' not in log:
                epoch_train_losses[epoch] = log['loss']
    
    # Convert to sorted lists based on epochs
    all_epochs = sorted(set(list(epoch_train_losses.keys()) + list(epoch_val_losses.keys())))
    
    for epoch in all_epochs:
        if epoch in epoch_train_losses:
            train_losses.append(epoch_train_losses[epoch])
        if epoch in epoch_val_losses:
            val_losses.append(epoch_val_losses[epoch])
        if epoch in epoch_val_accs:
            val_accs.append(epoch_val_accs[epoch])
    
    # If no epoch-level metrics found, try to extract from final metrics
    if not train_losses:
        if hasattr(train_result, 'training_loss'):
            train_losses = [train_result.training_loss]
    if not val_losses:
        if 'eval_loss' in eval_result:
            val_losses = [eval_result['eval_loss']]
    if not val_accs:
        if 'eval_accuracy' in eval_result:
            val_accs = [eval_result['eval_accuracy']]
    
    # Ensure train_losses and val_losses have the same length
    # Pad with last value if needed
    if train_losses and val_losses:
        max_len = max(len(train_losses), len(val_losses))
        # Pad shorter list with last value
        if len(train_losses) < max_len:
            train_losses.extend([train_losses[-1]] * (max_len - len(train_losses)))
        if len(val_losses) < max_len:
            val_losses.extend([val_losses[-1]] * (max_len - len(val_losses)))
        if val_accs and len(val_accs) < max_len:
            val_accs.extend([val_accs[-1]] * (max_len - len(val_accs)))
    
    # Ensure train_losses and val_losses have the same length
    # If they don't match, use the minimum length
    if train_losses and val_losses:
        min_len = min(len(train_losses), len(val_losses))
        train_losses = train_losses[:min_len]
        val_losses = val_losses[:min_len]
    
    # Detect overfitting
    overfitting_info = detect_overfitting(
        train_losses, val_losses, train_accs if train_accs else None, val_accs if val_accs else None
    )
    
    # Log overfitting analysis
    if overfitting_info['overfitting_detected']:
        logger.warning("⚠ Overfitting detected!")
        for msg in overfitting_info['analysis']:
            logger.warning(f"  {msg}")
    elif overfitting_info['underfitting_detected']:
        logger.warning("⚠ Underfitting detected!")
        for msg in overfitting_info['analysis']:
            logger.warning(f"  {msg}")
    else:
        logger.info("✓ No overfitting/underfitting detected")
    
    if overfitting_info['best_epoch'] is not None:
        logger.info(f"  Best epoch: {overfitting_info['best_epoch'] + 1}")
        logger.info(f"  Best validation loss: {overfitting_info['best_val_loss']:.4f}")
    
    # Save model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    
    # Generate predictions on validation set for monitoring
    logger.info("Generating validation predictions for monitoring...")
    val_predictions = trainer.predict(val_dataset)
    val_pred_proba = torch.softmax(torch.tensor(val_predictions.predictions), dim=1).numpy()[:, 1]
    val_pred_binary = np.argmax(val_predictions.predictions, axis=1)
    val_labels = np.array([val_dataset[i]['labels'].item() for i in range(len(val_dataset))])
    
    # Setup monitoring directory (similar to CatBoost/LightGBM)
    # Use same pattern as main.py: output/monitoring/camembert
    if visualization_dir is not None:
        # Try to infer project root from visualization_dir
        # visualization_dir is typically output/camembert/finetuning
        # So go up to output, then to monitoring/camembert
        if 'output' in str(visualization_dir):
            output_base = visualization_dir.parent.parent  # Go to output/
            monitoring_dir = output_base / "monitoring" / "camembert"
        else:
            # Fallback: use visualization_dir parent
            monitoring_dir = visualization_dir.parent / "monitoring" / "camembert"
    else:
        # Default fallback
        monitoring_dir = Path("output") / "monitoring" / "camembert"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization
    if save_plots and len(train_losses) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot training curves (save to both visualization_dir and monitoring_dir)
        plot_finetuning_curves(
            train_losses,
            val_losses,
            train_accs if train_accs else None,
            val_accs if val_accs else None,
            output_path=visualization_dir / f"training_curves_{timestamp}.png",
            overfitting_info=overfitting_info
        )
        
        # Save training curves to monitoring directory (standardized naming)
        _save_monitoring_plots(
            train_losses, val_losses, train_accs, val_accs,
            val_labels, val_pred_binary, val_pred_proba,
            monitoring_dir, overfitting_info
        )
        
        # Save training history and analysis
        history_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs if train_accs else [],
            'val_accs': val_accs if val_accs else [],
            'overfitting_analysis': overfitting_info,
            'final_metrics': {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy']
            },
            'training_config': {
                'model_name': model_name,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_length': max_length
            }
        }
        
        with open(visualization_dir / f"finetuning_history_{timestamp}.json", 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {visualization_dir}")
        logger.info(f"Monitoring plots saved to {monitoring_dir}")
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs if train_accs else [],
        'val_accs': val_accs if val_accs else [],
        'overfitting_info': overfitting_info
    }
    
    return model, tokenizer, training_history


def save_finetuned_model(
    model,
    tokenizer,
    output_dir: Path
) -> None:
    """
    Save fine-tuned model and tokenizer.
    保存微调后的模型和分词器。
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    logger.info(f"Fine-tuned model saved to {output_dir}")


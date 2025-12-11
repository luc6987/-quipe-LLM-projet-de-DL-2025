#!/usr/bin/python3
"""
DeBERTa Feature Extraction Fine-tuning Script
DeBERTa 特征提取微调脚本

This script fine-tunes `microsoft/deberta-v3-base` using supervised learning (with labels)
to extract the final-layer [CLS] representation as text embeddings for downstream XGBoost models.
The embeddings are saved in pickle format for easy integration with XGBoost pipelines.

Key features:
- Loads data from JSONL (both train and test sets).
- Uses Stratified Split to maintain label distribution for train/val split.
- Applies optional LoRA (Low-Rank Adaptation) or full-parameter fine-tuning.
- Uses Hugging Face `Trainer` with mixed precision (fp16).
- Supports Optuna for hyperparameter optimization.
- Extracts and saves [CLS] embeddings for both train and test sets.
- Saves fine-tuned model for future use.
"""

import os
import sys
import json
import argparse

# Add project root to path to allow imports from src
# This ensures the script can be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# From src/utils/bertweet_lora/ -> go up 3 levels to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add custom package path
# Keep site-packages from the virtualenv first; avoid overriding torch/numpy.

# Setup HuggingFace cache using unified path configuration
# MUST be set BEFORE importing transformers/huggingface_hub
from src.utils.paths import setup_huggingface_cache, PROJECT_ROOT, DATA_DIR, get_models_output_dir, get_embeddings_dir
setup_huggingface_cache()

import random
import inspect
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    DebertaV2Tokenizer
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
# Use unified path configuration
DATA_PATH = str(DATA_DIR / "train.jsonl")
TEST_DATA_PATH = str(DATA_DIR / "kaggle_test.jsonl")
OUTPUT_DIR = get_models_output_dir() / "bertweet_lora"
BEST_PARAMS_FILE = str(OUTPUT_DIR / "deberta_best_params.json")
EMBEDDING_DIR = str(get_embeddings_dir())

# Default Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query_proj", "key_proj", "value_proj", "o_proj"]
USE_LORA = True
CLS_DROPOUT = 0.1

LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512  # DeBERTa v3 supports up to 512 tokens. Can be adjusted based on needs.

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(pred: EvalPrediction):
    """Compute accuracy metric for the trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def str2bool(value):
    """Parse boolean-like CLI arguments."""
    if isinstance(value, bool):
        return value
    val = value.lower()
    if val in {"yes", "true", "t", "1", "y"}:
        return True
    if val in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from: {value}")


def infer_target_modules(model_name: str, fallback: Optional[List[str]] = None) -> List[str]:
    """Infer sensible LoRA target modules based on backbone name."""
    name = model_name.lower()
    if "deberta" in name:
        return ["query_proj", "key_proj", "value_proj", "o_proj"]
    if "roberta" in name or "bert" in name:
        return ["query", "key", "value", "dense"]
    if fallback:
        return fallback
    return TARGET_MODULES


class FocalLoss(nn.Module):
    """Simple focal loss for binary classification."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        focal_factor = (1 - prob) ** self.gamma
        loss = -(focal_factor * log_prob)
        loss = loss.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return loss.mean()


def symmetric_kl_with_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two logit tensors."""
    log_prob_a = F.log_softmax(logits_a, dim=-1)
    log_prob_b = F.log_softmax(logits_b, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    kl_ab = (prob_a * (log_prob_a - log_prob_b)).sum(dim=-1).mean()
    kl_ba = (prob_b * (log_prob_b - log_prob_a)).sum(dim=-1).mean()
    return 0.5 * (kl_ab + kl_ba)


class SafePEFTWrapper(nn.Module):
    """
    A wrapper around PEFT models that filters out 'labels' and 'label' parameters
    before passing them to the underlying model. This prevents errors when the
    base model doesn't accept these parameters.
    
    一个包装 PEFT 模型的包装器，在传递给底层模型之前过滤掉 'labels' 和 'label' 参数。
    这可以防止当基础模型不接受这些参数时出现错误。
    
    This is a one-time solution that works for all PEFT models regardless of
    how they internally handle parameters.
    这是一个一劳永逸的解决方案，适用于所有 PEFT 模型，无论它们内部如何处理参数。
    """
    def __init__(self, peft_model):
        super().__init__()
        # Use normal assignment to ensure peft_model is registered as a submodule by nn.Module
        # This ensures .to(device) works correctly
        self.peft_model = peft_model
        
        # Directly set config and base_model in __init__ to avoid recursion in __getattr__
        # 在 __init__ 中直接设置 config 和 base_model 以避免在 __getattr__ 中递归
        try:
            # Try to get config directly from __dict__ first
            peft_dict = peft_model.__dict__
            if 'config' in peft_dict:
                object.__setattr__(self, 'config', peft_dict['config'])
            else:
                # If not in __dict__, it might be a property, access it once
                object.__setattr__(self, 'config', peft_model.config)
        except (AttributeError, KeyError):
            # If config doesn't exist, set to None
            object.__setattr__(self, 'config', None)
        
        try:
            peft_dict = peft_model.__dict__
            if 'base_model' in peft_dict:
                object.__setattr__(self, 'base_model', peft_dict['base_model'])
            else:
                object.__setattr__(self, 'base_model', peft_model.base_model)
        except (AttributeError, KeyError):
            object.__setattr__(self, 'base_model', None)
    
    def forward(self, *args, **kwargs):
        # Create a clean copy of kwargs
        # Filter out known problematic arguments that might be passed by Trainer
        problematic_args = {"labels", "label", "num_items_in_batch"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_args}
        
        # Call the wrapped PEFT model with clean kwargs
        # Access peft_model directly from _modules to avoid triggering __getattr__
        modules = object.__getattribute__(self, '_modules')
        peft_model = modules['peft_model']
        
        return peft_model(*args, **clean_kwargs)
    
    def __getattr__(self, name):
        # First, check standard nn.Module attributes (_modules, _parameters, _buffers)
        # This mimics nn.Module's __getattr__ behavior
        if name in ('_modules', '_parameters', '_buffers'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Check if it's in _modules, _parameters, or _buffers (standard nn.Module behavior)
        # This should be handled by nn.Module, but we check here to be safe
        modules = object.__getattribute__(self, '_modules')
        parameters = object.__getattribute__(self, '_parameters')
        buffers = object.__getattribute__(self, '_buffers')
        
        if name in modules:
            return modules[name]
        if name in parameters:
            return parameters[name]
        if name in buffers:
            return buffers[name]
        
        # Check if it's one of our explicitly cached attributes
        # These should be accessed directly, not through __getattr__
        if name in ('config', 'base_model'):
            # These are set in __init__, so if we reach here, they don't exist
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Delegate all other attribute access to the wrapped PEFT model
        # Access peft_model directly from _modules to avoid recursion
        try:
            if 'peft_model' not in modules:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            peft_model = modules['peft_model']
            return getattr(peft_model, name)
        except (KeyError, AttributeError) as e:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from e


class CLSEmbeddingExtractor(nn.Module):
    """
    Embedding extractor that uses supervised learning to fine-tune the encoder.
    The classifier head is used for training, but the main purpose is to extract
    the final-layer [CLS] representation as text embeddings.
    """

    def __init__(
        self,
        encoder: AutoModel,
        num_labels: int = 2,
        cls_dropout: float = CLS_DROPOUT,
        loss_fn: Optional[nn.Module] = None,
        rdrop_alpha: float = 0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.num_labels = num_labels
        dropout_prob = cls_dropout if cls_dropout is not None else getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.rdrop_alpha = rdrop_alpha

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ) -> SequenceClassifierOutput:
        # Transformer models do not accept `labels`/`label` kwargs, so drop them
        # and keep only the tensors they carry.
        # Handle labels from both function signature and kwargs
        label_fallback = kwargs.pop("labels", None)
        single_label = kwargs.pop("label", None)
        if labels is None:
            labels = label_fallback if label_fallback is not None else single_label

        # Create a clean kwargs dict without problematic args to ensure they never reach the encoder
        # 创建一个干净的 kwargs 字典，不包含有问题的参数，确保它们永远不会到达 encoder
        problematic_args = {"labels", "label", "num_items_in_batch"}
        clean_encoder_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_args}
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **clean_encoder_kwargs
        )
        cls_repr = encoder_outputs.last_hidden_state[:, 0, :]
        logits_main = self.classifier(self.dropout(cls_repr))

        loss = None
        logits = logits_main
        if labels is not None:
            labels = labels.view(-1)
            if self.training and self.rdrop_alpha > 0:
                logits_aux = self.classifier(self.dropout(cls_repr))
                ce = (self.loss_fn(logits_main, labels) + self.loss_fn(logits_aux, labels)) * 0.5
                kl = symmetric_kl_with_logits(logits_main, logits_aux)
                loss = ce + self.rdrop_alpha * kl
                logits = (logits_main + logits_aux) * 0.5
            else:
                loss = self.loss_fn(logits_main, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )


def build_loss_fn(focal_gamma: float) -> nn.Module:
    if focal_gamma and focal_gamma > 0:
        return FocalLoss(gamma=focal_gamma)
    return nn.CrossEntropyLoss()


def build_text_encoder(
    model_name: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]] = None,
    gradient_checkpointing: bool = False
) -> AutoModel:
    """
    Build text encoder with optional LoRA adaptation.
    If use_lora=False, returns the base model for full-parameter fine-tuning.
    """
    config = AutoConfig.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, config=config)
    if gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    if use_lora:
        modules = target_modules or infer_target_modules(model_name)
        # Use FEATURE_EXTRACTION instead of SEQ_CLS because we are wrapping the encoder
        # and providing our own custom classification head. This prevents PEFT from
        # expecting/handling 'labels' internally.
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=modules
        )
        peft_model = get_peft_model(encoder, lora_config)
        peft_model.print_trainable_parameters()
        # Wrap PEFT model with SafePEFTWrapper to filter out labels parameter
        encoder = SafePEFTWrapper(peft_model)
    # If use_lora=False, return the base model directly for full-parameter fine-tuning
    return encoder


def collect_cls_embeddings(model, dataset, data_collator, batch_size):
    """
    Return CLS embeddings (and optional labels) for the given dataset.
    Returns numpy arrays for embeddings and labels (if available).
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    embeddings = []
    labels = []
    indices = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
            encoder_outputs = model.encoder(**batch_inputs)
            # Extract last layer [CLS] token representation
            cls_repr = encoder_outputs.last_hidden_state[:, 0, :].detach().cpu()
            embeddings.append(cls_repr)
            if "labels" in batch:
                labels.append(batch["labels"].clone().cpu())
            # Store batch indices for tracking
            batch_size_actual = cls_repr.shape[0]
            start_idx = idx * batch_size
            indices.extend(range(start_idx, start_idx + batch_size_actual))
    embedding_tensor = torch.cat(embeddings)
    label_array = torch.cat(labels).numpy() if labels else None
    indices_array = np.array(indices) if indices else None
    return embedding_tensor.numpy(), label_array, indices_array


def export_cls_embeddings(model, dataset, data_collator, batch_size, output_path: str):
    """
    Export CLS embeddings for a dataset to disk in pickle format.
    Format: {"embeddings": np.ndarray, "labels": np.ndarray (optional), "indices": np.ndarray (optional)}
    """
    import pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    embedding_array, label_array, indices_array = collect_cls_embeddings(model, dataset, data_collator, batch_size)
    output = {"embeddings": embedding_array}
    if label_array is not None:
        output["labels"] = label_array
    if indices_array is not None:
        output["indices"] = indices_array
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved CLS embeddings to {output_path}")
    print(f"  Embeddings shape: {embedding_array.shape}")
    if label_array is not None:
        print(f"  Labels shape: {label_array.shape}")


def generate_predictions(model, dataset, data_collator, batch_size):
    """
    Generate predictions for a dataset using the trained model.
    Returns numpy array of predicted class labels (0 or 1).
    """
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
            outputs = model(**batch_inputs)
            logits = outputs.logits
            # Get predicted class (argmax)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            predictions.append(preds)
    
    return np.concatenate(predictions)


def generate_submission(model, test_dataset, test_ids, data_collator, batch_size, output_path: str):
    """
    Generate submission file from test predictions.
    
    Args:
        model: Trained model
        test_dataset: Test dataset (tokenized)
        test_ids: Series of challenge IDs
        data_collator: Data collator for batching
        batch_size: Batch size for prediction
        output_path: Path to save submission CSV file
    """
    if test_dataset is None or test_ids is None:
        print("Warning: Cannot generate submission - test dataset or IDs not available.")
        return
    
    predictions = generate_predictions(model, test_dataset, data_collator, batch_size)
    
    if len(predictions) != len(test_ids):
        print(f"Error: Predictions length ({len(predictions)}) doesn't match IDs length ({len(test_ids)})")
        return
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'ID': test_ids.values,
        'Prediction': predictions
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"  Predictions distribution:")
    print(f"    {submission_df['Prediction'].value_counts().sort_index().to_dict()}")

# ---------------------------------------------------------------------------
# Data Loading & Processing
# ---------------------------------------------------------------------------
def get_tokenizer(model_name=MODEL_NAME):
    """Get tokenizer instance for DeBERTa models."""
    lower_name = model_name.lower()
    if "deberta" in lower_name:
        return DebertaV2Tokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)

def create_dataset_from_text(texts, labels=None, tokenizer=None, max_length=MAX_LENGTH):
    """
    Create Hugging Face Dataset from text list/series.
    Useful for integration with other scripts (e.g., stacking).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    # Ensure texts is a list or array, handling potential Series index issues
    if hasattr(texts, 'values'):
        texts = texts.values
    if labels is not None and hasattr(labels, 'values'):
        labels = labels.values
        
    df_data = {'text': texts}
    if labels is not None:
        df_data['label'] = labels
        
    df = pd.DataFrame(df_data)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length,
            padding=False
        )
        
    # Use map to tokenize
    # Note: map can be slow for very large datasets if not batched properly, but fine here
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Remove raw text
    if "text" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset

def load_and_process_data(train_data_path: str, test_data_path: Optional[str] = None):
    """
    Load JSONL data for both train and test sets, split train into train/val,
    and convert to Hugging Face Datasets.
    
    Returns:
        train_dataset, val_dataset, test_dataset, test_ids (test_dataset and test_ids can be None)
    """
    print(f"Loading training data from {train_data_path}...")
    try:
        train_df = pd.read_json(train_data_path, lines=True)
    except ValueError as e:
        print(f"Error loading training JSONL: {e}")
        sys.exit(1)
    
    # Ensure required columns exist for training data
    if 'text' not in train_df.columns or 'label' not in train_df.columns:
        print("Error: Training data must contain 'text' and 'label' columns.")
        sys.exit(1)
        
    # Keep only relevant columns to save memory
    train_df = train_df[['text', 'label']]
    
    print(f"Total training samples: {len(train_df)}")
    print(f"Label distribution:\n{train_df['label'].value_counts(normalize=True)}")

    # Stratified Split for train/val
    print("Splitting training data (80% Train, 20% Val)...")
    train_split_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        stratify=train_df['label'], 
        random_state=SEED
    )
    
    print(f"Train size: {len(train_split_df)}")
    print(f"Val size: {len(val_df)}")

    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(train_split_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    
    # Load test data if provided
    test_dataset = None
    test_ids = None
    if test_data_path and os.path.exists(test_data_path):
        print(f"\nLoading test data from {test_data_path}...")
        try:
            test_df = pd.read_json(test_data_path, lines=True)
            if 'text' not in test_df.columns:
                print("Warning: Test data must contain 'text' column. Skipping test data.")
            else:
                # Save challenge_id for submission generation
                if 'challenge_id' in test_df.columns:
                    test_ids = test_df['challenge_id'].copy()
                else:
                    # Fallback: use index as ID if challenge_id not found
                    print("Warning: 'challenge_id' not found in test data. Using index as ID.")
                    test_ids = pd.Series(range(len(test_df)), name='challenge_id')
                
                test_df = test_df[['text']]  # Test data doesn't have labels
                print(f"Test samples: {len(test_df)}")
                test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
        except Exception as e:
            print(f"Warning: Could not load test data: {e}. Continuing without test data.")
    
    return train_dataset, val_dataset, test_dataset, test_ids

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------
def train_model_with_params(
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    weight_decay: float,
    train_dataset,
    val_dataset,
    tokenizer,
    data_collator,
    output_dir: str = str(OUTPUT_DIR),
    use_full_data: bool = False,
    model_name: str = MODEL_NAME,
    use_lora: bool = USE_LORA,
    target_modules: Optional[List[str]] = None,
    focal_gamma: float = 0.0,
    rdrop_alpha: float = 0.0,
    gradient_checkpointing: bool = False
):
    """
    Train model with specified hyperparameters.
    Returns the trained trainer and final accuracy.
    
    Args:
        use_full_data: If True, train on full dataset (train+val) without validation split
    """
    encoder = build_text_encoder(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=gradient_checkpointing
    )
    loss_fn = build_loss_fn(focal_gamma)
    model = CLSEmbeddingExtractor(
        encoder=encoder,
        num_labels=2,
        cls_dropout=CLS_DROPOUT,
        loss_fn=loss_fn,
        rdrop_alpha=rdrop_alpha
    )
    
    if use_full_data:
        # For final model: train on all data, no validation split
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=weight_decay,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            eval_strategy="no",  # No validation during training
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            fp16=True,
            report_to="none",
            seed=SEED
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        
        # Evaluate on val set for reporting (but model was trained on full data)
        if val_dataset is not None:
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            accuracy = eval_result["eval_accuracy"]
        else:
            accuracy = 0.0  # No validation set available
    else:
        # For optimization: train with validation split
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=weight_decay,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,
            report_to="none",
            seed=SEED
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        accuracy = eval_result["eval_accuracy"]
    
    return trainer, accuracy

# ---------------------------------------------------------------------------
# Training Function (Objective for Optuna)
# ---------------------------------------------------------------------------
def objective(
    trial,
    train_dataset,
    val_dataset,
    tokenizer,
    data_collator,
    model_name: str = MODEL_NAME,
    use_lora: bool = USE_LORA,
    target_modules: Optional[List[str]] = None,
    focal_gamma: float = 0.0,
    rdrop_alpha: float = 0.0,
    gradient_checkpointing: bool = False
):
    """
    Optuna Objective Function
    """
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)

    # Use a unique output dir for each trial to avoid conflicts
    trial_output_dir = str(OUTPUT_DIR / f"trial_{trial.number}")
    
    # Train model using the shared training function
    trainer, accuracy = train_model_with_params(
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        batch_size=batch_size,
        weight_decay=weight_decay,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        output_dir=trial_output_dir,
        model_name=model_name,
        use_lora=use_lora,
        target_modules=target_modules,
        focal_gamma=focal_gamma,
        rdrop_alpha=rdrop_alpha,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # Clean up to save space (but keep the best model checkpoint)
    import shutil
    try:
        # Keep only the best checkpoint, remove others
        checkpoint_dirs = [d for d in os.listdir(trial_output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Find the best checkpoint (usually the last one due to load_best_model_at_end)
            best_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
            # We'll keep the entire trial dir for now, but could optimize later
            # For now, just remove logs to save space
            logs_dir = os.path.join(trial_output_dir, "logs")
            if os.path.exists(logs_dir):
                shutil.rmtree(logs_dir)
    except Exception as e:
        print(f"Warning: Could not clean up trial directory: {e}")

    return accuracy

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DeBERTa Feature Extraction Fine-tuning / Optimization")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--use_best_params", action="store_true", help="Load best parameters from JSON and retrain")
    parser.add_argument("--params_file", type=str, default=BEST_PARAMS_FILE, help="Path to best parameters JSON file")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Backbone model name (e.g., microsoft/deberta-v3-base)")
    parser.add_argument("--use_lora", type=str2bool, default=USE_LORA, help="Enable LoRA adapters (true/false)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None, help="Override LoRA target modules")
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Gamma for focal loss (0 disables)")
    parser.add_argument("--rdrop_alpha", type=float, default=0.0, help="R-Drop coefficient (0 disables)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for encoder")
    parser.add_argument("--export_embeddings", action="store_true", default=True, help="Export CLS embeddings for train/validation/test splits (default: True)")
    parser.add_argument("--embedding_dir", type=str, default=EMBEDDING_DIR, help="Directory for exported embeddings")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to test data JSONL file (optional)")
    parser.add_argument("--generate_submission", action="store_true", default=True, help="Generate submission file from test predictions (default: True)")
    parser.add_argument("--submission_path", type=str, default=None, help="Path to save submission CSV file (default: output/submissions/deberta_submission.csv)")
    args = parser.parse_args()

    # 1. Setup
    set_seed(SEED)
    model_name = args.model_name
    use_lora = args.use_lora
    target_modules = args.lora_target_modules
    focal_gamma = args.focal_gamma
    rdrop_alpha = args.rdrop_alpha
    gradient_checkpointing = args.gradient_checkpointing
    
    # Use unified path configuration for data paths
    from src.utils.paths import DATA_DIR
    train_data_path = str(DATA_DIR / "train.jsonl")
    test_data_path = args.test_data_path
    if test_data_path is None:
        test_data_path = str(DATA_DIR / "kaggle_test.jsonl")
    
    if not os.path.exists(train_data_path):
        print(f"Error: Training data file not found at {train_data_path}")
        sys.exit(1)

    # 2. Data Loading
    train_dataset, val_dataset, test_dataset, test_ids = load_and_process_data(train_data_path, test_data_path)

    # 3. Tokenization
    print(f"Initializing Tokenizer ({model_name})...")
    tokenizer = get_tokenizer(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_LENGTH,
            padding=False # Dynamic padding handled by DataCollator
        )

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Tokenize test dataset if available
    test_tokenized = None
    if test_dataset is not None:
        test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    # Remove raw text column to prepare for model
    train_tokenized = train_tokenized.remove_columns(["text"])
    val_tokenized = val_tokenized.remove_columns(["text"])
    if test_tokenized is not None:
        test_tokenized = test_tokenized.remove_columns(["text"])
    
    # Ensure format is torch
    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")
    if test_tokenized is not None:
        test_tokenized.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def maybe_export_embeddings(model, split_map, batch_size):
        """Export embeddings for all splits in split_map."""
        if not args.export_embeddings:
            return
        os.makedirs(args.embedding_dir, exist_ok=True)
        for split_name, dataset in split_map.items():
            if dataset is None:
                continue
            output_path = os.path.join(args.embedding_dir, f"{split_name}_embeddings.pkl")
            export_cls_embeddings(model, dataset, data_collator, batch_size, output_path)

    if args.optimize:
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna is not installed. Please install it to use --optimize.")
            sys.exit(1)
            
        study = optuna.create_study(direction="maximize")
        
        # Pass static arguments to objective using lambda or partial is tricky with pickle, 
        # but inside main() functions are local. We can define a wrapper.
        def objective_wrapper(trial):
            return objective(
                trial,
                train_tokenized,
                val_tokenized,
                tokenizer,
                data_collator,
                model_name=model_name,
                use_lora=use_lora,
                target_modules=target_modules,
                focal_gamma=focal_gamma,
                rdrop_alpha=rdrop_alpha,
                gradient_checkpointing=gradient_checkpointing
            )
            
        study.optimize(objective_wrapper, n_trials=args.n_trials)
        
        trial = study.best_trial
            
        # Save best params
        with open(BEST_PARAMS_FILE, "w") as f:
            json.dump(trial.params, f, indent=4)
        
        # Retrain with best parameters and save model
        # Use FULL training set (train + val) for final model
        best_lr = trial.params['learning_rate']
        best_r = trial.params['lora_r']
        best_alpha = trial.params['lora_alpha']
        best_dropout = trial.params['lora_dropout']
        best_batch_size = trial.params['batch_size']
        best_wd = trial.params['weight_decay']
        
        # Combine train and val for final training
        from datasets import concatenate_datasets
        full_train_tokenized = concatenate_datasets([train_tokenized, val_tokenized])
        
        trainer, final_accuracy = train_model_with_params(
            learning_rate=best_lr,
            lora_r=best_r,
            lora_alpha=best_alpha,
            lora_dropout=best_dropout,
            batch_size=best_batch_size,
            weight_decay=best_wd,
            train_dataset=full_train_tokenized,
            val_dataset=val_tokenized,  # Used only for final evaluation report
            tokenizer=tokenizer,
            data_collator=data_collator,
            output_dir=str(OUTPUT_DIR),
            use_full_data=True,  # Train on full dataset without validation split
            model_name=model_name,
            use_lora=use_lora,
            target_modules=target_modules,
            focal_gamma=focal_gamma,
            rdrop_alpha=rdrop_alpha,
            gradient_checkpointing=gradient_checkpointing
        )
        
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        
        # Export embeddings for all splits
        split_map = {
            "train": full_train_tokenized,
            "val": val_tokenized
        }
        if test_tokenized is not None:
            split_map["test"] = test_tokenized
        maybe_export_embeddings(trainer.model, split_map, best_batch_size)
        
        # Generate submission if requested
        if args.generate_submission and test_tokenized is not None and test_ids is not None:
            from src.utils.paths import get_submissions_dir
            if args.submission_path is None:
                submission_path = str(get_submissions_dir() / "deberta_submission.csv")
            else:
                submission_path = args.submission_path
            generate_submission(trainer.model, test_tokenized, test_ids, data_collator, best_batch_size, submission_path)
        
    elif args.use_best_params:
        # Load best parameters from JSON and retrain
        if not os.path.exists(args.params_file):
            print(f"Error: Parameters file not found: {args.params_file}")
            print(f"Please run with --optimize first to generate best parameters.")
            sys.exit(1)
        
        with open(args.params_file, "r") as f:
            best_params = json.load(f)
        
        best_lr = best_params['learning_rate']
        best_r = best_params['lora_r']
        best_alpha = best_params['lora_alpha']
        best_dropout = best_params['lora_dropout']
        best_batch_size = best_params['batch_size']
        best_wd = best_params['weight_decay']
        
        # Combine train and val for final training
        from datasets import concatenate_datasets
        full_train_tokenized = concatenate_datasets([train_tokenized, val_tokenized])
        
        trainer, final_accuracy = train_model_with_params(
            learning_rate=best_lr,
            lora_r=best_r,
            lora_alpha=best_alpha,
            lora_dropout=best_dropout,
            batch_size=best_batch_size,
            weight_decay=best_wd,
            train_dataset=full_train_tokenized,
            val_dataset=val_tokenized,  # Used only for final evaluation report
            tokenizer=tokenizer,
            data_collator=data_collator,
            output_dir=str(OUTPUT_DIR),
            use_full_data=True,  # Train on full dataset without validation split
            model_name=model_name,
            use_lora=use_lora,
            target_modules=target_modules,
            focal_gamma=focal_gamma,
            rdrop_alpha=rdrop_alpha,
            gradient_checkpointing=gradient_checkpointing
        )
        
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        
        # Export embeddings for all splits
        split_map = {
            "train": full_train_tokenized,
            "val": val_tokenized
        }
        if test_tokenized is not None:
            split_map["test"] = test_tokenized
        maybe_export_embeddings(trainer.model, split_map, best_batch_size)
        
        # Generate submission if requested
        if args.generate_submission and test_tokenized is not None and test_ids is not None:
            from src.utils.paths import get_submissions_dir
            if args.submission_path is None:
                submission_path = str(get_submissions_dir() / "deberta_submission.csv")
            else:
                submission_path = args.submission_path
            generate_submission(trainer.model, test_tokenized, test_ids, data_collator, best_batch_size, submission_path)
        
    else:
        # Standard Training (Manual Params)
        trainer, final_accuracy = train_model_with_params(
            learning_rate=LEARNING_RATE,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            output_dir=str(OUTPUT_DIR),
            model_name=model_name,
            use_lora=use_lora,
            target_modules=target_modules,
            focal_gamma=focal_gamma,
            rdrop_alpha=rdrop_alpha,
            gradient_checkpointing=gradient_checkpointing
        )

        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        
        # Export embeddings for all splits
        split_map = {
            "train": train_tokenized,
            "val": val_tokenized
        }
        if test_tokenized is not None:
            split_map["test"] = test_tokenized
        maybe_export_embeddings(trainer.model, split_map, BATCH_SIZE)
        
        # Generate submission if requested
        if args.generate_submission and test_tokenized is not None and test_ids is not None:
            from src.utils.paths import get_submissions_dir
            if args.submission_path is None:
                submission_path = str(get_submissions_dir() / "deberta_submission.csv")
            else:
                submission_path = args.submission_path
            generate_submission(trainer.model, test_tokenized, test_ids, data_collator, BATCH_SIZE, submission_path)

if __name__ == "__main__":
    main()

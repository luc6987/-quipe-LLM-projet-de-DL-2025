"""
L2 Stacking Ensemble Strategy
L2 堆叠集成策略

This script implements a 5-Fold Stratified Cross-Validation stacking strategy.
It generates Out-of-Fold (OOF) predictions for two base models (XGBoost and BERTweet),
trains a Logistic Regression meta-model, and produces final predictions on the test set.

Structure:
1. Data Loading (using src.utils.data_processing)
2. Mock Functions for Base Models (to be filled by user)
3. Stacking Class (manages OOF loop and meta-model)
4. Main Execution

防止数据泄露 (Data Leakage Prevention):
- Split data into folds BEFORE any model-specific processing (except basic cleaning).
- Train models strictly on X_train_fold.
- Predict on X_val_fold only after the model is trained on X_train_fold.
- Never use X_val_fold statistics/information during training of the current fold.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# =============================================================================
# Set Hugging Face cache directory using unified path configuration
# MUST be set BEFORE importing transformers/huggingface_hub
# =============================================================================
from src.utils.paths import setup_huggingface_cache, PROJECT_ROOT, DATA_DIR, get_submissions_dir
setup_huggingface_cache()

import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

from src.utils import data_processing as dp
from src.xgboost import xgboost_baseline
from src.bertweet_lora import train_lora

# =============================================================================
# Mock Functions (Placeholder for User Implementation)
# =============================================================================

import shutil
import gc

def train_predict_xgboost(train_x, train_y, val_x, val_y, test_x):
    """
    Train XGBoost model using src.xgboost.xgboost_baseline.
    Returns raw LOGITS (margin).
    """
    print("  [XGBoost] Training (via src.xgboost)...")
    
    # Use shared parameters
    params = xgboost_baseline.get_default_params()
    
    # Train using the shared function
    model = xgboost_baseline.train_single_model(
        train_x, train_y, 
        val_x, val_y, 
        params=params
    )
    
    # Predict Logits (output_margin=True)
    # Convert to numpy if needed (though predict handles DF usually)
    X_val_np = val_x.values if hasattr(val_x, 'values') else val_x
    X_test_np = test_x.values if hasattr(test_x, 'values') else test_x
    
    val_logits = model.predict(X_val_np, output_margin=True)
    test_logits = model.predict(X_test_np, output_margin=True)
    
    return val_logits, test_logits


def train_predict_bertweet(train_text, train_y, val_text, val_y, test_text):
    """
    Fine-tune the DeBERTa text encoder and return CLS embeddings for val/test.
    """
    print("  [DeBERTa] Fine-tuning + embedding extraction...")
    
    # Get Tokenizer
    tokenizer = train_lora.get_tokenizer()
    
    # Create Datasets using the shared utility
    train_tokenized = train_lora.create_dataset_from_text(train_text, train_y, tokenizer)
    val_tokenized = train_lora.create_dataset_from_text(val_text, val_y, tokenizer)
    test_tokenized = train_lora.create_dataset_from_text(test_text, None, tokenizer)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Use a temp dir for each fold
    fold_output_dir = f"./bertweet_lora_temp_fold"
    
    # Train
    # Using default parameters from train_lora constants
    trainer, _ = train_lora.train_model_with_params(
        learning_rate=train_lora.LEARNING_RATE,
        lora_r=train_lora.LORA_R,
        lora_alpha=train_lora.LORA_ALPHA,
        lora_dropout=train_lora.LORA_DROPOUT,
        batch_size=train_lora.BATCH_SIZE,
        weight_decay=train_lora.WEIGHT_DECAY,
        train_dataset=train_tokenized,
        val_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        output_dir=fold_output_dir
    )
    
    print("  [DeBERTa] Collecting CLS embeddings...")
    val_embeddings, _ = train_lora.collect_cls_embeddings(
        trainer.model, val_tokenized, data_collator, train_lora.BATCH_SIZE
    )
    test_embeddings, _ = train_lora.collect_cls_embeddings(
        trainer.model, test_tokenized, data_collator, train_lora.BATCH_SIZE
    )
    
    # Cleanup
    if os.path.exists(fold_output_dir):
        shutil.rmtree(fold_output_dir)
    
    # Force GPU memory cleanup to prevent OOM
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_embeddings, test_embeddings


# =============================================================================
# Stacking Logic
# =============================================================================

class StackingEnsemble:
    def __init__(self, n_folds=5, seed=42, text_components=128):
        self.n_folds = n_folds
        self.seed = seed
        self.text_components = text_components
        self.meta_model = LogisticRegression(random_state=seed, max_iter=1000)
        self.scaler = StandardScaler()
        self.reducer = TruncatedSVD(n_components=text_components, random_state=seed)
        self.oof_text_embeddings = None
        self.avg_test_text_embeddings = None
        
    def run_oof(self, train_df, test_df):
        """
        Execute the 5-Fold OOF loop to generate meta-features.
        
        核心逻辑：
        严禁使用 cross_val_predict，因为我们需要对深度学习模型进行自定义训练控制。
        我们需要手动切分 Fold，确保验证集 (Validation Set) 纯净，不参与当前 Fold 的训练。
        """
        print(f"\nStarting {self.n_folds}-Fold Stacking...")
        
        # Prepare data containers
        # OOF predictions (length of train data)
        oof_preds_a = np.zeros(len(train_df))
        test_preds_a_accum = np.zeros(len(test_df))
        text_embed_dim = None
        test_text_emb_accum = None
        
        # Labels for validation
        y = train_df['label'].values
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        # Extract features for Model A (Tabular) - using dp utils
        # Note: In a real scenario, you might want to do feature engineering inside the fold to be 100% safe,
        # but for tabular metadata (counts, ratios), doing it globally is usually acceptable if no target encoding is leaked.
        # Here we assume df already has engineered features or we generate them.
        # Let's assume we process them globally for simplicity as per common Kaggle practice for static features.
        print("Preparing features...")
        # Process text data (for Model B)
        # Assuming 'text' column exists or 'extended_tweet.full_text' etc.
        # We'll try to find a text column
        text_col = 'text'
        if text_col not in train_df.columns:
            raise ValueError("Expected 'text' column in training dataframe.")
            
        # Process tabular data (for Model A)
        # We use dp.engineer_features to get the tabular representation
        train_df_eng, transformers = dp.engineer_features(train_df, is_train=True)
        test_df_eng, _ = dp.engineer_features(test_df, is_train=False, transformers=transformers)
        
        # Prepare X, y for tabular
        X_tabular, _, _ = dp.prepare_xy(train_df_eng)
        X_test_tabular, _, _ = dp.prepare_xy(test_df_eng)
        
        # Align columns
        common_cols = [c for c in X_tabular.columns if c in X_test_tabular.columns]
        # Fill missing in test with 0
        missing_cols = set(X_tabular.columns) - set(X_test_tabular.columns)
        for c in missing_cols:
            X_test_tabular[c] = 0
        X_test_tabular = X_test_tabular[X_tabular.columns]
        
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tabular, y), 1):
            print(f"\nFold {fold}/{self.n_folds}")
            
            # --- Split Data ---
            # Tabular
            X_train_fold = X_tabular.iloc[train_idx]
            X_val_fold = X_tabular.iloc[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Text (Raw text for BERTweet)
            # We use the original DataFrame for text to avoid processing loss
            # Ensure we use the same indices
            train_text_fold = train_df.iloc[train_idx][text_col] if text_col in train_df else None
            val_text_fold = train_df.iloc[val_idx][text_col] if text_col in train_df else None
            test_text = test_df[text_col] if text_col in test_df else None
            
            # --- Model A: XGBoost ---
            print("  Model A (XGBoost)...")
            p_val_a, p_test_a = train_predict_xgboost(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                X_test_tabular
            )
            
            # Store predictions
            oof_preds_a[val_idx] = p_val_a
            test_preds_a_accum += p_test_a
            
            # --- Model B: BERTweet ---
            print("  Model B (BERTweet)...")
            if train_text_fold is not None:
                val_emb, test_emb = train_predict_bertweet(
                    train_text_fold, y_train_fold,
                    val_text_fold, y_val_fold,
                    test_text
                )
                if text_embed_dim is None:
                    text_embed_dim = val_emb.shape[1]
                    self.oof_text_embeddings = np.zeros((len(train_df), text_embed_dim), dtype=np.float32)
                    test_text_emb_accum = np.zeros((len(test_df), text_embed_dim), dtype=np.float32)
                self.oof_text_embeddings[val_idx] = val_emb
                test_text_emb_accum += test_emb
            else:
                raise ValueError("Text column is required for DeBERTa embeddings.")
        
        # Average test predictions
        self.avg_test_preds_a = test_preds_a_accum / self.n_folds
        self.avg_test_text_embeddings = test_text_emb_accum / self.n_folds
        
        return oof_preds_a, y

    def train_meta_model(self, oof_a, y):
        """
        Train the Logistic Regression Meta-Model.
        Uses XGBoost logits + reduced CLS embeddings as features.
        """
        print("\nTraining Meta-Model (Logistic Regression on Logits)...")
        
        if self.oof_text_embeddings is None:
            raise ValueError("Text embeddings are missing. Run run_oof() first.")
        embed_dim = self.oof_text_embeddings.shape[1]
        target_components = min(self.text_components, max(embed_dim - 1, 1))
        if self.reducer.n_components != target_components:
            self.reducer = TruncatedSVD(n_components=target_components, random_state=self.seed)
        reduced_text = self.reducer.fit_transform(self.oof_text_embeddings)
        self.current_components = target_components
        X_meta = np.column_stack([oof_a, reduced_text])
        X_scaled = self.scaler.fit_transform(X_meta)
        
        # Train Meta Model
        self.meta_model.fit(X_scaled, y)
        
        # Evaluate on OOF
        oof_meta_preds = self.meta_model.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        best_acc = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 81):
            acc = accuracy_score(y, oof_meta_preds > thresh)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        self.best_threshold = best_thresh
        acc = accuracy_score(y, oof_meta_preds > self.best_threshold)
        auc = roc_auc_score(y, oof_meta_preds)
        
        print(f"Meta-Model OOF Performance:")
        print(f"  Accuracy: {acc:.4f} (Threshold: {self.best_threshold:.4f})")
        print(f"  AUC:      {auc:.4f}")
        print(f"  Feature dims: XGB logit + {target_components}-D CLS projection")
            
        return acc, auc

    def predict_test(self):
        """
        Generate final test predictions using the Meta-Model.
        """
        print("\nGenerating Final Test Predictions...")
        
        if self.avg_test_text_embeddings is None:
            raise ValueError("Test text embeddings missing.")
        reduced_text = self.reducer.transform(self.avg_test_text_embeddings)
        X_meta_test = np.column_stack([self.avg_test_preds_a, reduced_text])
        X_meta_test = self.scaler.transform(X_meta_test)
        
        # Predict
        final_preds = self.meta_model.predict_proba(X_meta_test)[:, 1]
        
        return final_preds


def main():
    print("="*60)
    print("L2 Stacking Ensemble (XGBoost + BERTweet)")
    print("="*60)
    print(f"Hugging Face cache directory: {hf_cache_dir}")
    print("="*60)
    
    # 1. Load Data
    from src.utils.paths import DATA_DIR
    train_df = dp.load_data(str(DATA_DIR / 'train.jsonl'), clean_text=True)
    test_df = dp.load_data(str(DATA_DIR / 'kaggle_test.jsonl'), clean_text=True)
    
    # --- Better Text Column Handling ---
    print("Processing text columns (Merging extended and base text)...")
    
    def prepare_text(df):
        # 确保基础 text 列存在
        if 'text' not in df.columns:
            df['text'] = ""
            
        # 如果存在 extended_tweet.full_text，优先使用它填补
        if 'extended_tweet.full_text' in df.columns:
            # fillna: 如果 extended 也是 NaN，保持原 text
            # 注意：要先确保 text 是填充过的，避免覆盖
            df['text'] = df['extended_tweet.full_text'].combine_first(df['text'])
        
        # 再次检查 full_text (Twitter v2 format)
        if 'full_text' in df.columns:
             df['text'] = df['full_text'].combine_first(df['text'])
             
        # 最终清理：填充 NaN 为空字符串，并转为 str 类型
        df['text'] = df['text'].fillna("").astype(str)
        return df

    train_df = prepare_text(train_df)
    test_df = prepare_text(test_df)


    # 2. Initialize Stacker
    stacker = StackingEnsemble(n_folds=5, seed=42, text_components=128)
    
    # 3. Run OOF Loop
    oof_a, y_true = stacker.run_oof(train_df, test_df)
    
    # 4. Train Meta Model
    stacker.train_meta_model(oof_a, y_true)
    
    # 5. Predict Test
    final_preds = stacker.predict_test()
    
    # 6. Create Submission
    # Convert probabilities to binary predictions using optimal threshold
    binary_preds = (final_preds >= stacker.best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'ID': test_df['challenge_id'],
        'Prediction': binary_preds
    })
    
    output_file = str(get_submissions_dir() / 'submission_stacking.csv')
    submission.to_csv(output_file, index=False)
    print(f"\nSubmission saved to: {output_file}")


if __name__ == "__main__":
    main()


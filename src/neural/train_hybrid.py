"""
Training Script for Hybrid BERTweet + Tabular Model
混合模型训练脚本

Combines 194 tabular features with BERTweet text features.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import data_processing as dp
from src.neural.hybrid_model import HybridBERTweetTabular
from src.utils.paths import DATA_DIR, get_models_output_dir

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, tabular_cols=None, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
        # Text
        self.texts = df['text'].fillna("").astype(str).values
        
        # Tabular
        if tabular_cols:
            self.tabular = df[tabular_cols].values.astype(np.float32)
        else:
            self.tabular = np.zeros((len(df), 1), dtype=np.float32)
            
        # Labels
        if not is_test:
            self.labels = df['label'].values
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'tabular_features': torch.tensor(self.tabular[idx], dtype=torch.float)
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, n_examples):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        tabular_features = d["tabular_features"].to(device)
        targets = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tabular_features=tabular_features
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            tabular_features = d["tabular_features"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tabular_features=tabular_features
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

def main():
    # Config
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 2e-5
    MODEL_NAME = 'microsoft/deberta-v3-base'
    
    # Data
    train_df = dp.load_data(str(DATA_DIR / 'train.jsonl'), clean_text=True)
    
    # Ensure text column
    if 'text' not in train_df.columns:
        candidates = ['extended_tweet.full_text', 'full_text', 'text']
        for c in candidates:
            if c in train_df.columns:
                train_df['text'] = train_df[c]
                break
    
    # Feature Engineering (Tabular)
    logger.info("Processing features...")
    train_eng, transformers = dp.engineer_features(train_df, is_train=True)
    
    # Prepare Tabular Columns List
    # Exclude non-feature columns
    exclude = ['text', 'label', 'challenge_id', 'extended_tweet.full_text'] 
    tabular_cols = [c for c in train_eng.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_eng[c])]
    logger.info(f"Using {len(tabular_cols)} tabular features.")
    
    # Split
    train_df_split, val_df_split = train_test_split(train_eng, test_size=0.2, random_state=42, stratify=train_eng['label'])
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = HybridDataset(train_df_split, tokenizer, MAX_LEN, tabular_cols)
    val_dataset = HybridDataset(val_df_split, tokenizer, MAX_LEN, tabular_cols)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridBERTweetTabular(
        model_name=MODEL_NAME,
        num_tabular_features=len(tabular_cols)
    )
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Loop
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, len(train_df_split)
        )
        print(f"Train loss {train_loss} accuracy {train_acc}")
        
        val_acc, val_loss = eval_model(
            model, val_loader, loss_fn, device, len(val_df_split)
        )
        print(f"Val   loss {val_loss} accuracy {val_acc}")
        
        if val_acc > best_accuracy:
            model_path = str(get_models_output_dir() / 'hybrid_model' / 'best_hybrid_model.bin')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            best_accuracy = val_acc
            
    print("Training Complete.")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class HybridBERTweetTabular(nn.Module):
    """
    Hybrid Model combining BERTweet (Text) and MLP (Tabular).
    
    Architecture:
    - Text Branch: BERTweet -> Pooler Output ([CLS])
    - Tabular Branch: MLP -> Embedding
    - Fusion: Concatenation
    - Head: Classifier
    """
    def __init__(self, 
                 model_name='vinai/bertweet-base', 
                 num_tabular_features=194, 
                 tabular_hidden_dim=64, 
                 num_labels=2,
                 dropout_rate=0.3):
        super().__init__()
        
        # 1. Text Branch (BERTweet)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. Tabular Branch (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, tabular_hidden_dim),
            nn.BatchNorm1d(tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. Classifier Head
        bert_hidden_size = self.config.hidden_size
        combined_dim = bert_hidden_size + tabular_hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, tabular_features):
        """
        Args:
            input_ids (torch.Tensor): [batch_size, seq_len]
            attention_mask (torch.Tensor): [batch_size, seq_len]
            tabular_features (torch.Tensor): [batch_size, num_tabular_features]
            
        Returns:
            logits (torch.Tensor): [batch_size, num_labels]
        """
        # Text Branch
        # outputs.last_hidden_state: [batch, seq, dim]
        # outputs.pooler_output: [batch, dim] (processed [CLS])
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooler_output if available, else mean pooling or CLS
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            text_embedding = outputs.pooler_output
        else:
            # Fallback to CLS token (first token)
            text_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Tabular Branch
        tab_embedding = self.tabular_mlp(tabular_features)
        
        # Fusion
        combined = torch.cat((text_embedding, tab_embedding), dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits




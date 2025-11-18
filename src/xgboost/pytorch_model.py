"""
PyTorch神经网络模型：可选的PyTorch神经网络模型 / PyTorch neural network model: Optional PyTorch neural network model
适合表格数据的多层全连接网络，支持GPU训练 / Multi-layer fully connected network suitable for tabular data, supports GPU training
"""

import os
import sys
# 添加 /Data 目录中的包路径（如果存在） / Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from .pytorch_data_loader import get_device, create_data_loader


class TabularNet(nn.Module):
    """
    表格数据神经网络模型 / Tabular data neural network model
    多层全连接网络，适合表格数据分类任务 / Multi-layer fully connected network suitable for tabular data classification tasks
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, 
                 activation='relu', num_classes=1):
        """
        Args:
            input_dim: 输入特征维度 / Input feature dimension
            hidden_dims: 隐藏层维度列表 / Hidden layer dimension list
            dropout: Dropout比率 / Dropout ratio
            activation: 激活函数（'relu'或'gelu'） / Activation function ('relu' or 'gelu')
            num_classes: 输出类别数（1表示二分类） / Number of output classes (1 for binary classification)
        """
        super(TabularNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(model, train_loader, val_loader=None, epochs=50, 
                       lr=0.001, device=None, verbose=True):
    """
    训练PyTorch模型
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据DataLoader
        val_loader: 验证数据DataLoader（可选）
        epochs: 训练轮数
        lr: 学习率
        device: 设备（如果为None，自动检测）
        verbose: 是否打印训练信息
        
    Returns:
        dict: 包含训练历史的字典
    """
    if device is None:
        device, _ = get_device()
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).unsqueeze(1)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def predict_pytorch_model(model, data_loader, device=None):
    """
    使用PyTorch模型进行预测
    
    Args:
        model: 训练好的PyTorch模型
        data_loader: 数据DataLoader
        device: 设备（如果为None，自动检测）
        
    Returns:
        numpy数组: 预测概率
    """
    if device is None:
        device, _ = get_device()
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch_x in data_loader:
            if isinstance(batch_x, tuple):
                batch_x = batch_x[0]
            
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions).flatten()
    return predictions


def create_pytorch_model_for_tabular(X_train, y_train, X_val=None, y_val=None,
                                     hidden_dims=[256, 128, 64], dropout=0.3,
                                     epochs=50, lr=0.001, batch_size=1024,
                                     device=None, verbose=True):
    """
    创建并训练PyTorch表格数据模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        hidden_dims: 隐藏层维度列表
        dropout: Dropout比率
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批量大小
        device: 设备（如果为None，自动检测）
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含模型和训练历史的字典
    """
    if device is None:
        device, use_gpu = get_device()
    else:
        use_gpu = device.type == 'cuda'
    
    # 获取输入维度
    if isinstance(X_train, pd.DataFrame):
        input_dim = X_train.shape[1]
    else:
        input_dim = X_train.shape[1]
    
    # 创建模型
    model = TabularNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes=1
    )
    
    if verbose:
        print(f"\nCreating PyTorch Model:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Dropout: {dropout}")
        print(f"  Device: {device}")
    
    # 创建数据加载器
    train_loader = create_data_loader(
        X_train, y_train, batch_size=batch_size, 
        shuffle=True, device=device
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = create_data_loader(
            X_val, y_val, batch_size=batch_size,
            shuffle=False, device=device
        )
    
    # 训练模型
    history = train_pytorch_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, device=device, verbose=verbose
    )
    
    return {
        'model': model,
        'history': history,
        'device': device
    }


"""
PyTorch数据加载模块：使用PyTorch进行GPU加速的数据预处理和加载 / PyTorch data loading module: GPU-accelerated data preprocessing and loading using PyTorch
支持大规模特征矩阵的GPU操作和批量处理 / Supports GPU operations and batch processing for large-scale feature matrices
"""

import os
import sys
# 添加 /Data 目录中的包路径（如果存在） / Add /Data directory package paths (if they exist)
if os.path.exists('/Data/yuguang.yao/pytorch_packages'):
    sys.path.insert(0, '/Data/yuguang.yao/pytorch_packages')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


def get_device():
    """
    检测并返回可用的设备（GPU或CPU） / Detect and return available device (GPU or CPU)
    
    Returns:
        torch.device: 设备对象 / Device object
        bool: 是否使用GPU / Whether using GPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device, True
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
        return device, False


class TabularDataset(Dataset):
    """
    表格数据Dataset类，用于PyTorch DataLoader / Tabular data Dataset class for PyTorch DataLoader
    """
    
    def __init__(self, X, y=None, device=None):
        """
        Args:
            X: 特征DataFrame或numpy数组 / Feature DataFrame or numpy array
            y: 标签Series或numpy数组（可选） / Label Series or numpy array (optional)
            device: 设备（如果为None，自动检测） / Device (if None, auto-detect)
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            self.X = X.astype(np.float32)
        else:
            raise ValueError(f"Unsupported X type: {type(X)}")
        
        if y is not None:
            if isinstance(y, pd.Series):
                self.y = y.values.astype(np.float32)
            elif isinstance(y, np.ndarray):
                self.y = y.astype(np.float32)
            else:
                raise ValueError(f"Unsupported y type: {type(y)}")
        else:
            self.y = None
        
        self.device = device if device is not None else torch.device('cpu')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).to(self.device)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32).to(self.device)
            return x, y
        else:
            return x


def create_data_loader(X, y=None, batch_size=1024, shuffle=False, 
                      num_workers=0, pin_memory=True, device=None):
    """
    创建PyTorch DataLoader
    
    Args:
        X: 特征DataFrame或numpy数组
        y: 标签Series或numpy数组（可选）
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载工作进程数
        pin_memory: 是否使用pin_memory（加速GPU传输）
        device: 设备（如果为None，自动检测）
        
    Returns:
        DataLoader对象
    """
    if device is None:
        device, _ = get_device()
    
    dataset = TabularDataset(X, y, device=device)
    
    # 如果使用GPU，pin_memory可以加速数据传输
    if device.type == 'cuda' and pin_memory:
        pin_memory = True
    else:
        pin_memory = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


def preprocess_on_gpu(X, operations=None, batch_size=10000):
    """
    在GPU上预处理大规模特征矩阵
    
    支持的操作：
    - 标准化
    - 归一化
    - 缺失值填充
    - 特征变换
    
    Args:
        X: 特征DataFrame或numpy数组
        operations: 操作列表（目前主要用于演示，实际XGBoost不需要标准化）
        batch_size: 批量处理大小
        
    Returns:
        numpy数组: 预处理后的特征
    """
    device, use_gpu = get_device()
    
    # 转换为numpy数组
    if isinstance(X, pd.DataFrame):
        X_array = X.values.astype(np.float32)
        feature_names = X.columns.tolist()
    else:
        X_array = X.astype(np.float32)
        feature_names = None
    
    if use_gpu and len(X_array) > batch_size:
        # 分批处理（避免GPU内存不足）
        print(f"  Using GPU batch preprocessing (batch_size={batch_size})...")
        
        results = []
        for i in range(0, len(X_array), batch_size):
            batch = X_array[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)
            
            # 应用操作（示例：这里可以添加各种预处理操作）
            # 注意：XGBoost通常不需要标准化，这里仅作演示
            
            # 转换回numpy
            batch_processed = batch_tensor.cpu().numpy()
            results.append(batch_processed)
        
        X_processed = np.vstack(results)
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    else:
        # 小数据或CPU模式，直接处理
        if use_gpu:
            X_tensor = torch.from_numpy(X_array).to(device)
            X_processed = X_tensor.cpu().numpy()
            torch.cuda.empty_cache()
        else:
            X_processed = X_array
    
    # 转换回DataFrame（如果原来是DataFrame）
    if feature_names is not None:
        return pd.DataFrame(X_processed, columns=feature_names, index=X.index if hasattr(X, 'index') else None)
    else:
        return X_processed


def convert_to_tensor(X, device=None):
    """
    将DataFrame或numpy数组转换为PyTorch张量
    
    Args:
        X: 特征DataFrame或numpy数组
        device: 设备（如果为None，自动检测）
        
    Returns:
        torch.Tensor: PyTorch张量
    """
    if device is None:
        device, _ = get_device()
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values.astype(np.float32)
    elif isinstance(X, np.ndarray):
        X_array = X.astype(np.float32)
    else:
        raise ValueError(f"Unsupported X type: {type(X)}")
    
    return torch.from_numpy(X_array).to(device)


def convert_from_tensor(tensor, feature_names=None, index=None):
    """
    将PyTorch张量转换回DataFrame或numpy数组
    
    Args:
        tensor: PyTorch张量
        feature_names: 特征名称列表（如果提供，返回DataFrame）
        index: 索引（如果提供，返回DataFrame）
        
    Returns:
        DataFrame或numpy数组
    """
    # 移动到CPU并转换为numpy
    if tensor.is_cuda:
        array = tensor.cpu().numpy()
    else:
        array = tensor.numpy()
    
    if feature_names is not None:
        return pd.DataFrame(array, columns=feature_names, index=index)
    else:
        return array


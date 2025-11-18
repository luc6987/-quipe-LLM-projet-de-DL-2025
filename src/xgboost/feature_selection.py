"""
特征选择模块：实现指南第四部分的特征选择 / Feature selection module: Implements feature selection from guide part 4
包括XGBoost重要性分析、SelectFromModel和SHAP分析 / Includes XGBoost importance analysis, SelectFromModel and SHAP analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ Warning: shap not installed, cannot perform SHAP analysis")


def get_feature_importance(model, feature_names, importance_type='gain'):
    """
    获取XGBoost特征重要性 / Get XGBoost feature importance
    
    根据指南，应该使用importance_type='gain'（增益），而不是'weight'（频率） / According to guide, should use importance_type='gain' (gain), not 'weight' (frequency)
    
    Args:
        model: 训练好的XGBoost模型 / Trained XGBoost model
        feature_names: 特征名称列表 / Feature name list
        importance_type: 重要性类型（'gain'或'weight'） / Importance type ('gain' or 'weight')
        
    Returns:
        DataFrame: 包含特征名称和重要性的DataFrame / DataFrame containing feature names and importance
    """
    importances = model.get_booster().get_score(importance_type=importance_type)
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': [importances.get(f'f{i}', 0.0) for i in range(len(feature_names))]
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def select_features_from_model(model, X_train, y_train, threshold='median', 
                               importance_type='gain', prefit=True):
    """
    使用SelectFromModel进行特征选择
    
    Args:
        model: 训练好的XGBoost模型
        X_train: 训练特征
        y_train: 训练标签
        threshold: 重要性阈值（'median', 'mean', 或数值）
        importance_type: 重要性类型
        prefit: 模型是否已经训练好
        
    Returns:
        SelectFromModel对象
        DataFrame: 选择的特征
    """
    print("\n" + "="*60)
    print("Feature Selection: SelectFromModel")
    print("="*60)
    
    # 设置模型的重要性类型 / Set model importance type
    if hasattr(model, 'get_booster'):
        # XGBoost模型 / XGBoost model
        model.get_booster().set_param({'importance_type': importance_type})
    
    selector = SelectFromModel(
        model,
        threshold=threshold,
        prefit=prefit,
        importance_getter=lambda model: model.feature_importances_
    )
    
    if not prefit:
        selector.fit(X_train, y_train)
    
    # 获取选择的特征 / Get selected features
    selected_features = selector.transform(X_train)
    selected_mask = selector.get_support()
    
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    selected_feature_names = [name for name, mask in zip(feature_names, selected_mask) if mask]
    
    print(f"  Original feature count: {len(feature_names)}")
    print(f"  Selected feature count: {len(selected_feature_names)}")
    print(f"  Threshold: {threshold}")
    print(f"  Importance type: {importance_type}")
    
    return selector, pd.DataFrame(selected_features, columns=selected_feature_names)


def analyze_with_shap(model, X_val, feature_names=None, plot=True, 
                      max_display=20, output_dir=None):
    """
    使用SHAP进行特征可解释性分析 / Use SHAP for feature interpretability analysis
    
    根据指南第四部分，生成SHAP summary plot（beeswarm图） / According to guide part 4, generate SHAP summary plot (beeswarm plot)
    
    Args:
        model: 训练好的XGBoost模型 / Trained XGBoost model
        X_val: 验证集特征（用于SHAP分析） / Validation set features (for SHAP analysis)
        feature_names: 特征名称列表 / Feature name list
        plot: 是否生成可视化 / Whether to generate visualization
        max_display: 显示的最大特征数 / Maximum number of features to display
        output_dir: 输出目录（用于保存图片） / Output directory (for saving images)
        
    Returns:
        dict: 包含SHAP值的字典 / Dictionary containing SHAP values
            - 'shap_values': SHAP值数组 / SHAP values array
            - 'feature_importance': 特征重要性DataFrame / Feature importance DataFrame
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed, cannot perform SHAP analysis. Please install using pip install shap.")
    
    print("\n" + "="*60)
    print("SHAP Analysis: Feature Interpretability")
    print("="*60)
    
    # 转换为numpy数组 / Convert to numpy array
    if hasattr(X_val, 'values'):
        X_val_array = X_val.values
    else:
        X_val_array = X_val
    
    # 限制样本数（SHAP计算可能很慢） / Limit sample count (SHAP calculation may be slow)
    max_samples = min(1000, len(X_val_array))
    if len(X_val_array) > max_samples:
        print(f"  Using first {max_samples} samples for SHAP analysis (to speed up calculation)")
        X_val_array = X_val_array[:max_samples]
    
    # 创建SHAP解释器 / Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值 / Calculate SHAP values
    print("  Calculating SHAP values...")
    shap_values = explainer.shap_values(X_val_array)
    
    # 如果shap_values是列表（多类），取第一个 / If shap_values is list (multi-class), take first
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # 计算特征重要性（平均绝对SHAP值） / Calculate feature importance (mean absolute SHAP values)
    if feature_names is None:
        if hasattr(X_val, 'columns'):
            feature_names = X_val.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X_val_array.shape[1])]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\n  Top {min(max_display, len(feature_importance))} important features (by mean |SHAP value|):")
    for idx, row in feature_importance.head(max_display).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # 生成可视化 / Generate visualization
    if plot:
        try:
            print("\n  Generating SHAP visualization...")
            
            # Summary plot (beeswarm)
            shap.summary_plot(
                shap_values,
                X_val_array,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            if output_dir:
                import os
                import matplotlib.pyplot as plt
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), 
                           bbox_inches='tight', dpi=150)
                print(f"  ✓ SHAP summary plot saved to: {output_dir}/shap_summary_plot.png")
                plt.close()
            else:
                import matplotlib.pyplot as plt
                plt.show()
            
            # Bar plot（特征重要性） / Feature importance
            shap.plots.bar(
                shap_values,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), 
                           bbox_inches='tight', dpi=150)
                print(f"  ✓ SHAP bar plot saved to: {output_dir}/shap_bar_plot.png")
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"  ⚠ Warning: Error generating SHAP visualization: {e}")
    
    print(f"{'='*60}")
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'explainer': explainer
    }


def create_feature_importance_table(model, X_train, feature_names=None, 
                                   importance_type='gain'):
    """
    创建特征重要性表（用于最终报告）
    
    根据指南表3，创建可读的特征重要性摘要
    
    Args:
        model: 训练好的XGBoost模型
        X_train: 训练特征（用于获取特征名）
        feature_names: 特征名称列表（如果提供）
        importance_type: 重要性类型
        
    Returns:
        DataFrame: 特征重要性表
    """
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    importance_df = get_feature_importance(model, feature_names, importance_type)
    
    # 添加排名
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    # 重新排列列顺序
    importance_df = importance_df[['rank', 'feature', 'importance']]
    
    return importance_df


"""
XGBoost训练和预测脚本
完整流程：特征提取 -> 数据清洗 -> 模型训练 -> 预测
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: optuna 未安装，无法进行超参数优化。使用 pip install optuna 安装。")

# 导入特征提取和清洗模块（将在main函数中动态导入）


def safe_read_csv(file_path, chunk_size=None, max_size_mb=1000):
    """
    安全地读取CSV文件，支持大文件和错误处理
    
    Args:
        file_path: CSV文件路径
        chunk_size: 分块读取大小（行数），None表示一次性读取
        max_size_mb: 如果文件超过此大小（MB），使用分块读取
        
    Returns:
        DataFrame: 读取的数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 检查文件大小
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  读取文件: {file_path} (大小: {file_size_mb:.2f} MB)")
    
    try:
        if chunk_size is None and file_size_mb > max_size_mb:
            print("  文件较大，使用分块读取...")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ 读取完成: {df.shape}")
            return df
        elif chunk_size is not None:
            print(f"  使用分块读取 (chunk_size={chunk_size})...")
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            print(f"  ✓ 读取完成: {df.shape}")
            return df
        else:
            df = pd.read_csv(file_path)
            print(f"  ✓ 读取完成: {df.shape}")
            return df
    except Exception as e:
        print(f"  错误: 读取文件时出错: {str(e)}")
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            print("  提示: 文件读取超时，尝试使用分块读取...")
            try:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                print(f"  ✓ 分块读取成功: {df.shape}")
                return df
            except Exception as e2:
                raise IOError(f"分块读取也失败: {str(e2)}")
        else:
            raise


def load_labels_and_ids(train_labels_file='cache/train_labels.csv',
                       test_ids_file='cache/test_ids.csv'):
    """
    加载训练集标签和测试集ID
    
    Args:
        train_labels_file: 训练集标签文件路径
        test_ids_file: 测试集ID文件路径
        
    Returns:
        labels: 训练集标签Series
        test_ids: 测试集ID Series
    """
    print("\n" + "="*60)
    print("加载标签和ID")
    print("="*60)
    
    # 加载标签
    if os.path.exists(train_labels_file):
        try:
            labels_df = safe_read_csv(train_labels_file, max_size_mb=100)
            labels = labels_df['label']
            print(f"✓ 加载训练集标签: {len(labels)} 条")
        except Exception as e:
            print(f"错误: 读取标签文件失败: {str(e)}")
            labels = None
    else:
        print(f"错误: 找不到标签文件 {train_labels_file}")
        labels = None
    
    # 加载测试集ID
    if os.path.exists(test_ids_file):
        try:
            test_ids_df = safe_read_csv(test_ids_file, max_size_mb=100)
            test_ids = test_ids_df['challenge_id']
            print(f"✓ 加载测试集ID: {len(test_ids)} 条")
        except Exception as e:
            print(f"错误: 读取测试集ID文件失败: {str(e)}")
            test_ids = None
    else:
        print(f"警告: 找不到测试集ID文件 {test_ids_file}")
        test_ids = None
    
    return labels, test_ids


def train_xgboost_model(X_train, y_train, cv_folds=5, xgb_params=None):
    """
    训练XGBoost模型并进行交叉验证
    
    Args:
        X_train: 训练特征DataFrame
        y_train: 训练标签Series
        cv_folds: 交叉验证折数
        xgb_params: XGBoost参数字典，如果为None则使用默认参数
        
    Returns:
        model: 训练好的最终模型
        cv_scores: 交叉验证分数列表
    """
    print("\n" + "="*60)
    print("训练XGBoost模型")
    print("="*60)
    
    # 转换为numpy数组
    X_train_array = X_train.values
    y_train_array = y_train.values
    
    # 设置5折分层交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 默认XGBoost参数（改进版，提升性能）
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 7,  # 增加深度以捕获更复杂的模式
            'learning_rate': 0.05,  # 降低学习率，配合更多树
            'n_estimators': 500,  # 增加树的数量
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,  # 添加列采样层级
            'min_child_weight': 1,  # 降低最小子权重，允许更细的分割
            'gamma': 0.0,  # 降低gamma，允许更多分割
            'reg_alpha': 0.0,  # 降低L1正则化
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # 更快的训练方法
            'scale_pos_weight': 1,  # 如果类别不平衡，可以调整
            'max_delta_step': 0,  # 用于不平衡数据
        }
    
    print("\nXGBoost参数:")
    for key, value in xgb_params.items():
        print(f"  {key}: {value}")
    
    # 交叉验证
    print(f"\n运行 {cv_folds}折分层交叉验证...")
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_array, y_train_array), 1):
        print(f"\nFold {fold}/{cv_folds}:")
        
        X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
        y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
        
        # 创建XGBoost模型
        model = xgb.XGBClassifier(**xgb_params)
        
        # 训练
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )
        
        # 预测和评估
        y_pred = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, y_pred)
        cv_scores.append(acc)
        
        print(f"  验证准确率: {acc:.4f}")
    
    print(f"\n{'='*60}")
    print("交叉验证结果:")
    print(f"  平均准确率: {np.mean(cv_scores):.4f} ({np.mean(cv_scores)*100:.2f}%)")
    print(f"  标准差: {np.std(cv_scores):.4f} ({np.std(cv_scores)*100:.2f}%)")
    print(f"  各折分数: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"{'='*60}")
    
    # 训练最终模型（使用全部训练数据）
    print("\n使用全部训练数据训练最终模型...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_train_array, y_train_array)
    print("训练完成.")
    
    return final_model, cv_scores


def optimize_hyperparameters(X_train, y_train, cv_folds=5, n_trials=50, timeout=None):
    """
    使用Optuna进行XGBoost超参数优化
    
    Args:
        X_train: 训练特征DataFrame
        y_train: 训练标签Series
        cv_folds: 交叉验证折数
        n_trials: Optuna优化试验次数
        timeout: 优化超时时间（秒），None表示不限制
        
    Returns:
        best_params: 最佳参数字典
        best_score: 最佳交叉验证分数
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna 未安装，无法进行超参数优化。请使用 pip install optuna 安装。")
    
    print("\n" + "="*60)
    print("使用Optuna进行超参数优化")
    print("="*60)
    print(f"优化试验次数: {n_trials}")
    if timeout:
        print(f"超时时间: {timeout} 秒")
    
    X_train_array = X_train.values
    y_train_array = y_train.values
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def objective(trial):
        """Optuna优化目标函数"""
        # 定义超参数搜索空间
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # 交叉验证
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train_array, y_train_array):
            X_fold_train, X_fold_val = X_train_array[train_idx], X_train_array[val_idx]
            y_fold_train, y_fold_val = y_train_array[train_idx], y_train_array[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_fold_val)
            acc = accuracy_score(y_fold_val, y_pred)
            cv_scores.append(acc)
        
        return np.mean(cv_scores)
    
    # 创建Optuna study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 运行优化
    print("\n开始优化...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # 输出最佳参数
    print(f"\n{'='*60}")
    print("优化完成!")
    print(f"{'='*60}")
    print(f"最佳交叉验证准确率: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    print("\n最佳参数:")
    best_params = study.best_params.copy()
    # 添加固定参数
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    })
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
    
    return best_params, study.best_value


def make_predictions(model, X_test, test_ids, output_file='output/xgboost_baseline.csv'):
    """
    生成预测结果并保存为提交文件
    
    Args:
        model: 训练好的模型
        X_test: 测试特征DataFrame
        test_ids: 测试集ID Series
        output_file: 输出文件路径
        
    Returns:
        submission: 提交DataFrame
    """
    print("\n" + "="*60)
    print("生成预测结果")
    print("="*60)
    
    X_test_array = X_test.values
    predictions = model.predict(X_test_array)
    
    # 创建提交文件（格式：ID, Prediction）
    submission = pd.DataFrame({
        'ID': test_ids,
        'Prediction': predictions
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # 保存提交文件
    submission.to_csv(output_file, index=False)
    print(f"\n✓ 预测结果已保存到: {output_file}")
    print(f"  预测样本数: {len(predictions)}")
    print("  预测值分布:")
    print(submission['Prediction'].value_counts().sort_index())
    
    return submission


def main():
    """
    主函数：执行完整的训练和预测流程
    """
    import argparse
    
    # 获取项目根目录（从 src/xgboost 向上两级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 切换到项目根目录
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    # 将src/xgboost目录添加到Python路径，以便导入模块
    xgboost_dir = os.path.join(project_root, 'src', 'xgboost')
    if xgboost_dir not in sys.path:
        sys.path.insert(0, xgboost_dir)
    
    # 导入特征提取和清洗模块
    from extract_features import extract_all_features, extract_labels_and_ids
    from clean import clean_and_reduce
    
    try:
        parser = argparse.ArgumentParser(description='XGBoost训练和预测脚本')
        parser.add_argument('--train-file', type=str, default='data/train.jsonl',
                            help='训练集JSONL文件路径')
        parser.add_argument('--test-file', type=str, default='data/kaggle_test.jsonl',
                            help='测试集JSONL文件路径')
        parser.add_argument('--run-extract', action='store_true', default=False,
                            help='运行特征提取步骤（默认跳过，使用已存在的特征文件）')
        parser.add_argument('--run-clean', action='store_true', default=False,
                            help='运行数据清洗步骤（默认跳过，使用已存在的清洗文件）')
        parser.add_argument('--cv-folds', type=int, default=5,
                            help='交叉验证折数')
        parser.add_argument('--output', type=str, default='output/xgboost_baseline.csv',
                            help='输出提交文件路径')
        parser.add_argument('--optimize', action='store_true', default=False,
                            help='使用Optuna进行超参数优化（推荐用于提升性能）')
        parser.add_argument('--n-trials', type=int, default=100,
                            help='Optuna优化试验次数（仅在--optimize时有效，默认100）')
        parser.add_argument('--timeout', type=int, default=None,
                            help='Optuna优化超时时间（秒），None表示不限制（仅在--optimize时有效）')
        parser.add_argument('--quick-optimize', action='store_true', default=False,
                            help='快速优化模式：使用Optuna但只运行20次试验（用于快速测试）')
        
        args = parser.parse_args()
        
        print("="*60)
        print("XGBoost完整训练和预测流程")
        print("="*60)
        print(f"项目根目录: {project_root}")
        
        # ========== 步骤1: 特征提取 ==========
        if args.run_extract:
            print("\n[步骤1] 特征提取")
            print("="*60)
            
            # 提取训练集特征和标签
            train_features_df, train_feature_cols, _ = extract_all_features(
                args.train_file,
                'cache/extracted_features.csv',
                exclude_label=True,
                exclude_id=True
            )
            
            # 提取训练集标签
            labels, _ = extract_labels_and_ids(
                args.train_file,
                output_labels_file='cache/train_labels.csv'
            )
            
            # 提取测试集特征和ID
            test_features_df, test_feature_cols, test_challenge_ids = extract_all_features(
                args.test_file,
                'cache/test_extracted_features.csv',
                exclude_label=True,
                exclude_id=False
            )
            
            # 保存测试集ID（extract_all_features已经分离了challenge_id）
            if test_challenge_ids is not None:
                ids_df = pd.DataFrame({'challenge_id': test_challenge_ids})
                ids_df.to_csv('cache/test_ids.csv', index=False)
                print("✓ 测试集ID已保存到 cache/test_ids.csv")
        else:
            print("\n[步骤1] 跳过特征提取（使用已有文件）")
            labels = None
            test_challenge_ids = None
        
        # ========== 步骤2: 数据清洗 ==========
        if args.run_clean:
            print("\n[步骤2] 数据清洗和降维")
            print("="*60)
            
            df_train_cleaned, transformers, df_test_cleaned = clean_and_reduce(
                input_file='cache/extracted_features.csv',
                output_file='cache/cleaned_features.csv',
                missing_strategy='median',
                outlier_method='clip',
                remove_low_var=False,  # 禁用低方差特征移除，对XGBoost树模型通常不需要
                remove_high_corr=False,  # 禁用高相关特征移除，XGBoost可以处理相关特征
                normalize_method='none',  # 禁用标准化，XGBoost树模型不需要标准化
                reduce_method='none',  # 禁用降维，保留所有特征
                variance_ratio=0.95,
                test_input_file='cache/test_extracted_features.csv',
                test_output_file='cache/test_cleaned_features.csv'
            )
        else:
            print("\n[步骤2] 跳过数据清洗（使用已有文件）")
        
        # ========== 步骤3: 加载数据 ==========
        print("\n[步骤3] 加载清洗后的数据和标签")
        print("="*60)
        
        # 检查必需的文件是否存在
        train_cleaned_file = 'cache/cleaned_features.csv'
        test_cleaned_file = 'cache/test_cleaned_features.csv'
        
        if not os.path.exists(train_cleaned_file):
            print(f"错误: 找不到训练集清洗文件 {train_cleaned_file}")
            print("请先运行数据清洗步骤（使用 --run-clean）")
            return
        
        if not os.path.exists(test_cleaned_file):
            print(f"警告: 找不到测试集清洗文件 {test_cleaned_file}")
            print("正在自动运行数据清洗步骤以生成测试集清洗文件...")
            
            # 检查测试集提取文件是否存在
            test_extracted_file = 'cache/test_extracted_features.csv'
            if not os.path.exists(test_extracted_file):
                print(f"错误: 找不到测试集提取文件 {test_extracted_file}")
                print("请先运行特征提取步骤（使用 --run-extract）")
                return
            
            # 运行清洗步骤（仅处理测试集，使用已有的训练集清洗文件）
            print("从训练集清洗文件加载transformer信息...")
            # 由于我们需要完整的transformer信息，需要重新运行清洗
            # 但可以只读取训练集已清洗的数据，然后只处理测试集
            try:
                df_train_cleaned = safe_read_csv(train_cleaned_file)
                print(f"已加载训练集清洗数据: {df_train_cleaned.shape}")
            except Exception as e:
                print(f"错误: 无法读取训练集清洗文件: {str(e)}")
                return
            
            print("警告: 需要重新运行完整的数据清洗流程以正确处理测试集")
            print("请运行: python train_xgboost.py --run-clean")
            return
        
        # 加载清洗后的特征
        print("\n加载训练集特征...")
        X_train = safe_read_csv(train_cleaned_file)
        print("\n加载测试集特征...")
        X_test = safe_read_csv(test_cleaned_file)
        
        print(f"训练集特征形状: {X_train.shape}")
        print(f"测试集特征形状: {X_test.shape}")
        
        # 加载标签和ID
        if labels is None:
            labels, test_ids = load_labels_and_ids()
        else:
            test_ids = None
            if test_challenge_ids is not None:
                test_ids = pd.Series(test_challenge_ids)
            elif os.path.exists('cache/test_ids.csv'):
                try:
                    test_ids_df = safe_read_csv('cache/test_ids.csv', max_size_mb=100)
                    test_ids = test_ids_df['challenge_id']
                except Exception as e:
                    print(f"错误: 读取测试集ID文件失败: {str(e)}")
                    test_ids = None
        
        if labels is None:
            print("错误: 无法加载训练集标签")
            return
        
        if test_ids is None:
            print("错误: 无法加载测试集ID")
            return
        
        # 确保标签和特征数量匹配
        if len(labels) != len(X_train):
            print(f"警告: 标签数量 ({len(labels)}) 与特征数量 ({len(X_train)}) 不匹配")
            # 取较小值
            min_len = min(len(labels), len(X_train))
            labels = labels[:min_len]
            X_train = X_train.iloc[:min_len]
            print(f"已调整到: {min_len} 条")
        
        # 确保测试集ID和特征数量匹配
        # 注意：测试集不应该修改行数，所以应该以ID数量为准
        if len(test_ids) != len(X_test):
            print(f"警告: 测试集ID数量 ({len(test_ids)}) 与特征数量 ({len(X_test)}) 不匹配")
            if len(test_ids) > len(X_test):
                # 如果ID数量更多，说明特征数据被截断了，需要补充
                print(f"  特征数据行数不足，将用最后一行填充缺失的行")
                missing_rows = len(test_ids) - len(X_test)
                # 用最后一行填充
                last_row = X_test.iloc[-1:].copy()
                for _ in range(missing_rows):
                    X_test = pd.concat([X_test, last_row], ignore_index=True)
                print(f"  已补充 {missing_rows} 行，现在特征数据: {len(X_test)} 行")
            else:
                # 如果ID数量更少，截断特征数据以匹配ID
                print(f"  特征数据行数过多，将截断以匹配ID数量")
                X_test = X_test.iloc[:len(test_ids)]
                print(f"  已截断到 {len(test_ids)} 行")
            
            # 确保test_ids是Series以便正确使用
            if not isinstance(test_ids, pd.Series):
                test_ids = pd.Series(test_ids)
        
        # 确保测试集特征列与训练集一致
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        missing_cols = train_cols - test_cols
        extra_cols = test_cols - train_cols
        
        if missing_cols:
            print(f"  测试集缺少列: {len(missing_cols)} 个，用0填充")
            for col in missing_cols:
                X_test[col] = 0
        
        if extra_cols:
            print(f"  测试集多余列: {len(extra_cols)} 个，已移除")
            X_test = X_test.drop(columns=list(extra_cols))
        
        # 确保列顺序一致
        X_test = X_test[X_train.columns]
        
        print("✓ 数据加载完成")
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")
        
        # ========== 步骤4: 模型训练 ==========
        print("\n[步骤4] 模型训练")
        
        # 超参数优化
        xgb_params = None
        use_optimize = args.optimize or args.quick_optimize
        n_trials_to_use = 20 if args.quick_optimize else args.n_trials
        
        if use_optimize:
            if not OPTUNA_AVAILABLE:
                print("错误: optuna 未安装，无法进行超参数优化。")
                print("请使用 pip install optuna 安装，或移除 --optimize 参数。")
                print("将使用默认参数继续训练...")
            else:
                print(f"\n{'='*60}")
                print(f"开始超参数优化（试验次数: {n_trials_to_use}）")
                print(f"{'='*60}")
                best_params, best_score = optimize_hyperparameters(
                    X_train, labels, 
                    cv_folds=args.cv_folds,
                    n_trials=n_trials_to_use,
                    timeout=args.timeout
                )
                xgb_params = best_params
                print("\n使用优化后的参数进行最终训练...")
        else:
            print("\n提示: 可以使用 --optimize 或 --quick-optimize 进行超参数优化以提升性能")
            print("  --optimize: 完整优化（默认100次试验，较慢但效果好）")
            print("  --quick-optimize: 快速优化（20次试验，较快但效果一般）")
        
        model, cv_scores = train_xgboost_model(X_train, labels, cv_folds=args.cv_folds, xgb_params=xgb_params)
        
        # ========== 步骤5: 生成预测 ==========
        print("\n[步骤5] 生成预测")
        submission = make_predictions(model, X_test, test_ids, output_file=args.output)
        
        # ========== 步骤6: 特征重要性 ==========
        print("\n" + "="*60)
        print("Top 20 重要特征")
        print("="*60)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "="*60)
        print("流程完成!")
        print("="*60)
        print(f"提交文件: {args.output}")
        print(f"交叉验证平均准确率: {np.mean(cv_scores):.4f}")
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()


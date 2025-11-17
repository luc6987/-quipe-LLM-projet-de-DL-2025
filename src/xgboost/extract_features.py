"""
提取train.jsonl和test.jsonl数据中所有194个特征
将嵌套JSON展平为表格格式，排除文本特征
确保训练集和测试集有统一的特征列
"""

import json
import argparse
from pandas import json_normalize
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_all_feature_columns(train_file, test_file):
    """
    扫描训练集和测试集，获取所有可能的特征列
    
    Args:
        train_file: 训练集JSONL文件路径
        test_file: 测试集JSONL文件路径
        
    Returns:
        list: 所有特征列的列表（排序后）
    """
    print("正在扫描训练集和测试集以确定所有特征列...")
    
    all_columns = set()
    
    # 需要排除的文本列
    exclude_cols = {
        'text',
        'extended_tweet.full_text',
        'quoted_status.extended_tweet.full_text',
        'quoted_status.text',
        'label',  # 训练集的标签列
        'challenge_id'  # ID列（需要单独处理）
    }
    
    # 扫描训练集
    if train_file:
        try:
            print(f"  扫描训练集: {train_file}")
            with open(train_file, 'r', encoding='utf-8') as f:
                sample_records = []
                for i, line in enumerate(f):
                    if i < 100:  # 只读取前100行来获取列结构
                        try:
                            data = json.loads(line.strip())
                            sample_records.append(data)
                        except json.JSONDecodeError:
                            continue
                    else:
                        break
            
            if sample_records:
                df_sample = json_normalize(sample_records)
                train_cols = set(df_sample.columns) - exclude_cols
                all_columns.update(train_cols)
                print(f"    训练集特征数: {len(train_cols)}")
        except Exception as e:
            print(f"    警告: 扫描训练集时出错: {e}")
    
    # 扫描测试集
    if test_file:
        try:
            print(f"  扫描测试集: {test_file}")
            with open(test_file, 'r', encoding='utf-8') as f:
                sample_records = []
                for i, line in enumerate(f):
                    if i < 100:  # 只读取前100行来获取列结构
                        try:
                            data = json.loads(line.strip())
                            sample_records.append(data)
                        except json.JSONDecodeError:
                            continue
                    else:
                        break
            
            if sample_records:
                df_sample = json_normalize(sample_records)
                test_cols = set(df_sample.columns) - exclude_cols
                all_columns.update(test_cols)
                print(f"    测试集特征数: {len(test_cols)}")
        except Exception as e:
            print(f"    警告: 扫描测试集时出错: {e}")
    
    print(f"  总特征数（去重后）: {len(all_columns)}")
    return sorted(list(all_columns))  # 返回排序后的列名列表，确保顺序一致


def extract_all_features(input_file, output_file='extracted_features.csv',
                         exclude_label=True, exclude_id=True,
                         unified_feature_columns=None):
    """
    从JSONL文件中提取所有表格特征（排除文本列）
    使用统一的特征列列表，确保训练集和测试集特征一致
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的CSV文件路径
        exclude_label: 是否排除label列（默认True）
        exclude_id: 是否排除challenge_id列（默认True）
        unified_feature_columns: 统一的特征列列表（如果提供，确保输出包含所有列）
    """
    print(f"正在读取文件: {input_file}")
    print("正在展平JSON数据...")
    
    # 逐行读取并展平，更节省内存
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0 and i > 0:
                print(f"  已处理 {i} 行...")
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError as e:
                print(f"  警告: 第 {i+1} 行JSON解析失败: {e}")
                continue
    
    print(f"  成功加载 {len(records)} 条记录")
    print("  正在标准化JSON结构...")
    
    # 展平嵌套JSON
    df = json_normalize(records)
    
    print(f"  原始列数: {len(df.columns)}")
    print(f"  数据形状: {df.shape}")
    
    # 需要排除的文本列
    exclude_cols = [
        'text',
        'extended_tweet.full_text',
        'quoted_status.extended_tweet.full_text',
        'quoted_status.text'
    ]
    
    # 根据参数决定是否排除label和challenge_id
    if exclude_label and 'label' in df.columns:
        exclude_cols.append('label')
    if exclude_id and 'challenge_id' in df.columns:
        exclude_cols.append('challenge_id')
    
    # 获取当前文件中的特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 如果提供了统一的特征列列表，使用它来确保列的一致性
    if unified_feature_columns is not None:
        print(f"  使用统一的特征列列表（共 {len(unified_feature_columns)} 列）")
        
        # 创建包含所有统一特征列的DataFrame
        features_df = pd.DataFrame(index=df.index)
        
        # 填充存在的列
        for col in unified_feature_columns:
            if col in df.columns:
                features_df[col] = df[col]
            else:
                # 如果列不存在，用NaN填充
                features_df[col] = None
        
        # 确保列顺序与统一列表一致
        features_df = features_df[unified_feature_columns]
        
        # 统计信息
        existing_cols = set(feature_cols) & set(unified_feature_columns)
        missing_cols = set(unified_feature_columns) - set(feature_cols)
        print(f"    存在的特征: {len(existing_cols)}")
        if len(missing_cols) > 0:
            print(f"    缺失的特征（用NaN填充）: {len(missing_cols)}")
    else:
        # 使用当前文件中的特征列
        features_df = df[feature_cols].copy()
    
    # 如果challenge_id在特征中但需要分离，先提取它
    challenge_id_series = None
    if 'challenge_id' in features_df.columns and not exclude_id:
        challenge_id_series = features_df['challenge_id'].copy()
        # 从特征中移除challenge_id（如果exclude_id为False，说明我们需要保留它但需要分离）
        # 但如果exclude_id为False，说明我们不需要排除它，所以保留在特征中
        # 实际上，如果exclude_id=False，challenge_id应该在feature_cols中
        # 我们在这里不排除它，但在保存时可以选择是否保存
    
    print(f"\n提取的特征数量: {len(features_df.columns)}")
    print(f"特征数据形状: {features_df.shape}")
    
    # 显示特征列表（只显示前50个，避免输出过长）
    print("\n所有特征列名 (前50个):")
    display_cols = list(features_df.columns)[:50]
    for i, col in enumerate(display_cols, 1):
        print(f"  {i:3d}. {col}")
    if len(features_df.columns) > 50:
        print(f"  ... (共 {len(features_df.columns)} 个特征)")
    
    # 保存为CSV（如果exclude_id=False且challenge_id在特征中，保存时不包括challenge_id）
    # 因为challenge_id需要单独保存
    save_df = features_df.copy()
    if 'challenge_id' in save_df.columns and challenge_id_series is not None:
        # challenge_id将被单独处理，所以从保存的特征中移除
        save_df = save_df.drop(columns=['challenge_id'])
        feature_cols_for_save = [
            col for col in features_df.columns if col != 'challenge_id'
        ]
    else:
        feature_cols_for_save = list(features_df.columns)
    
    print(f"\n正在保存到: {output_file}")
    save_df.to_csv(output_file, index=False)
    print(f"✓ 成功保存 {len(feature_cols_for_save)} 个特征到 {output_file}")
    
    # 显示一些统计信息
    print("\n数据统计:")
    print(f"  总行数: {len(features_df)}")
    print(f"  总列数: {len(feature_cols_for_save)}")
    print(f"  缺失值总数: {save_df.isnull().sum().sum()}")
    if len(feature_cols_for_save) > 0:
        missing_ratio = (save_df.isnull().sum().sum() /
                         (len(save_df) * len(feature_cols_for_save)) * 100)
        print(f"  缺失值比例: {missing_ratio:.2f}%")

    # 显示每列的数据类型
    print("\n数据类型统计:")
    dtype_counts = save_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")
    
    return features_df, list(feature_cols_for_save), challenge_id_series


def extract_labels_and_ids(input_file, output_labels_file=None, output_ids_file=None):
    """
    从训练集或测试集中提取标签和ID
    
    Args:
        input_file: 输入的JSONL文件路径
        output_labels_file: 输出标签文件路径（可选，仅用于训练集）
        output_ids_file: 输出ID文件路径（可选，仅用于测试集）
        
    Returns:
        labels: 标签Series（如果有）或None
        ids: challenge_id Series（如果有）或None
    """
    print(f"\n正在提取标签和ID: {input_file}")
    
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError:
                continue
    
    df = json_normalize(records)
    
    labels = None
    ids = None
    
    if 'label' in df.columns:
        labels = df['label']
        if output_labels_file:
            labels.to_csv(output_labels_file, index=False, header=['label'])
            print(f"✓ 标签已保存到: {output_labels_file}")
    
    if 'challenge_id' in df.columns:
        ids = df['challenge_id']
        if output_ids_file:
            ids.to_csv(output_ids_file, index=False, header=['challenge_id'])
            print(f"✓ ID已保存到: {output_ids_file}")
    
    return labels, ids


def main():
    """
    主函数：支持命令行参数处理训练集和测试集
    确保训练集和测试集有统一的特征列
    """
    import os
    
    # 获取项目根目录（从 src/xgboost 向上两级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 切换到项目根目录
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        parser = argparse.ArgumentParser(description='从JSONL文件中提取特征')
        parser.add_argument(
            '--train-file', type=str, default='data/train.jsonl',
            help='训练集JSONL文件路径'
        )
        parser.add_argument(
            '--test-file', type=str, default='data/kaggle_test.jsonl',
            help='测试集JSONL文件路径'
        )
        parser.add_argument(
            '--train-output', type=str,
            default='cache/extracted_features.csv',
            help='训练集特征输出文件路径'
        )
        parser.add_argument(
            '--test-output', type=str,
            default='cache/test_extracted_features.csv',
            help='测试集特征输出文件路径'
        )
        parser.add_argument(
            '--train-labels-output', type=str,
            default='cache/train_labels.csv',
            help='训练集标签输出文件路径'
        )
        parser.add_argument(
            '--test-ids-output', type=str, default='cache/test_ids.csv',
            help='测试集ID输出文件路径'
        )
        parser.add_argument('--train-only', action='store_true',
                            help='仅处理训练集')
        parser.add_argument('--test-only', action='store_true',
                            help='仅处理测试集')
        
        args = parser.parse_args()
        
        print(f"项目根目录: {project_root}")
        print("="*60)
        print("提取训练集和测试集特征（统一特征列）")
        print("="*60)
        
        # 第一步：确定统一的特征列列表
        unified_columns = None
        if not args.train_only and not args.test_only:
            # 同时处理训练集和测试集时，先确定统一的特征列
            train_file_for_scan = (
                args.train_file if not args.test_only else None
            )
            test_file_for_scan = (
                args.test_file if not args.train_only else None
            )
            unified_columns = get_all_feature_columns(
                train_file_for_scan, test_file_for_scan
            )
            if unified_columns:
                print(f"\n✓ 已确定统一的特征列列表（共 {len(unified_columns)} 列）")
        
        # 处理训练集
        if not args.test_only:
            try:
                print("\n[处理训练集]")
                features_df, feature_cols, _ = extract_all_features(
                    args.train_file,
                    args.train_output,
                    exclude_label=True,  # 排除label列（只提取特征）
                    exclude_id=True,     # 排除challenge_id列
                    unified_feature_columns=unified_columns
                )
                
                # 提取并保存标签
                labels, _ = extract_labels_and_ids(
                    args.train_file,
                    output_labels_file=args.train_labels_output
                )
                
                print("\n✓ 训练集处理完成!")
                print(f"  特征文件: {args.train_output}")
                if labels is not None:
                    print(f"  标签文件: {args.train_labels_output}")
                
                # 如果还没有统一的列列表，使用训练集的列作为统一列表
                if unified_columns is None:
                    unified_columns = feature_cols
            except FileNotFoundError:
                print(f"错误: 找不到训练集文件 {args.train_file}")
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 处理测试集
        if not args.train_only:
            try:
                print("\n[处理测试集]")
                features_df, feature_cols, challenge_id_series = (
                    extract_all_features(
                        args.test_file,
                        args.test_output,
                        exclude_label=True,  # 测试集没有label列
                        exclude_id=False,    # 保留challenge_id（需要分离）
                        unified_feature_columns=unified_columns
                    )
                )
                
                # 保存challenge_id
                if challenge_id_series is not None:
                    ids_df = pd.DataFrame(
                        {'challenge_id': challenge_id_series}
                    )
                    ids_df.to_csv(args.test_ids_output, index=False)
                    print(
                        f"✓ challenge_id已保存到: {args.test_ids_output}"
                    )
                
                print("\n✓ 测试集处理完成!")
                print(f"  特征文件: {args.test_output}")
                if challenge_id_series is not None:
                    print(f"  ID文件: {args.test_ids_output}")
            except FileNotFoundError:
                print(f"错误: 找不到测试集文件 {args.test_file}")
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("全部提取完成!")
        print("="*60)
        if unified_columns:
            print(f"训练集和测试集已统一为 {len(unified_columns)} 个特征列")
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

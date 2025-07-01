#!/usr/bin/env python3
"""
生成示例数据用于测试冠军挑战者系统
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse
from loguru import logger


def generate_classification_data(n_samples=10000, n_features=20, n_classes=2, random_state=42):
    """
    生成分类数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        random_state: 随机种子
        
    Returns:
        DataFrame: 生成的数据
    """
    logger.info(f"Generating classification data: {n_samples} samples, {n_features} features, {n_classes} classes")
    
    # 生成基础数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.01  # 添加一些噪声
    )
    
    # 创建特征名称
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # 添加一些分类特征
    np.random.seed(random_state)
    
    # 添加分类特征
    categories_1 = ['A', 'B', 'C', 'D']
    df['category_1'] = np.random.choice(categories_1, size=n_samples)
    
    categories_2 = ['Type1', 'Type2', 'Type3']
    df['category_2'] = np.random.choice(categories_2, size=n_samples)
    
    # 添加一些文本特征（简化版）
    text_options = [
        'This is a sample text for feature engineering',
        'Another example of text data for processing',
        'Text feature with different content',
        'Sample description for model training',
        'Feature text with various characteristics'
    ]
    df['text_feature'] = np.random.choice(text_options, size=n_samples)
    
    # 添加一些时间相关特征
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 删除timestamp列（保留派生特征）
    df = df.drop('timestamp', axis=1)
    
    logger.info(f"Generated data shape: {df.shape}")
    return df


def add_data_drift(df, drift_strength=0.1):
    """
    添加数据漂移以模拟真实场景
    
    Args:
        df: 原始数据
        drift_strength: 漂移强度
        
    Returns:
        DataFrame: 带有漂移的数据
    """
    logger.info(f"Adding data drift with strength: {drift_strength}")
    
    df_drift = df.copy()
    
    # 对数值特征添加漂移
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'target']
    
    for col in numeric_columns:
        # 添加系统性偏移
        shift = np.random.normal(0, drift_strength * df[col].std())
        df_drift[col] = df_drift[col] + shift
        
        # 添加方差变化
        scale_factor = 1 + np.random.normal(0, drift_strength)
        df_drift[col] = df_drift[col] * scale_factor
    
    # 对分类特征添加分布变化
    if 'category_1' in df.columns:
        # 改变分类分布
        original_dist = df['category_1'].value_counts(normalize=True)
        new_probs = original_dist.values * (1 + np.random.normal(0, drift_strength, len(original_dist)))
        new_probs = new_probs / new_probs.sum()  # 归一化
        
        df_drift['category_1'] = np.random.choice(
            original_dist.index, 
            size=len(df_drift), 
            p=new_probs
        )
    
    logger.info("Data drift added successfully")
    return df_drift


def save_datasets(df, output_dir, prefix="sample_data"):
    """
    保存数据集
    
    Args:
        df: 数据
        output_dir: 输出目录
        prefix: 文件前缀
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 分割数据
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])
    
    # 保存文件
    datasets = {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'full': df
    }
    
    file_paths = {}
    for name, data in datasets.items():
        file_path = output_path / f"{prefix}_{name}.csv"
        data.to_csv(file_path, index=False)
        file_paths[name] = str(file_path)
        logger.info(f"Saved {name} data: {file_path} (shape: {data.shape})")
    
    return file_paths


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate sample data for MLOps testing")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--features", type=int, default=20, help="Number of features")
    parser.add_argument("--classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--generate-drift", action="store_true", help="Generate drift dataset")
    parser.add_argument("--drift-strength", type=float, default=0.1, help="Drift strength")
    
    args = parser.parse_args()
    
    logger.info("Starting sample data generation")
    logger.info(f"Parameters: samples={args.samples}, features={args.features}, classes={args.classes}")
    
    # 生成基础数据
    df_base = generate_classification_data(
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes
    )
    
    # 保存基础数据集
    base_paths = save_datasets(df_base, args.output_dir, "baseline")
    
    # 如果需要，生成漂移数据
    if args.generate_drift:
        logger.info("Generating drift dataset")
        df_drift = add_data_drift(df_base, drift_strength=args.drift_strength)
        drift_paths = save_datasets(df_drift, args.output_dir, "drift")
        
        logger.info("Drift dataset generated:")
        for name, path in drift_paths.items():
            logger.info(f"  {name}: {path}")
    
    logger.info("Sample data generation completed")
    logger.info("Base dataset files:")
    for name, path in base_paths.items():
        logger.info(f"  {name}: {path}")
    
    # 生成数据摘要
    logger.info("\nData Summary:")
    logger.info(f"Shape: {df_base.shape}")
    logger.info(f"Target distribution:\n{df_base['target'].value_counts()}")
    logger.info(f"Missing values: {df_base.isnull().sum().sum()}")
    
    # 显示特征信息
    logger.info("\nFeature types:")
    logger.info(f"Numeric features: {len(df_base.select_dtypes(include=[np.number]).columns)}")
    logger.info(f"Categorical features: {len(df_base.select_dtypes(include=['object']).columns)}")


if __name__ == "__main__":
    main()

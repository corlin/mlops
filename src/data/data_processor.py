"""
数据处理模块 - 负责数据预处理和特征工程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataProcessor:
    """数据处理器，负责数据的加载、清洗、特征工程和分割"""
    
    def __init__(self, config_path: str):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.scalers = {}
        self.encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
            
        Returns:
            清洗后的数据
        """
        logger.info("Starting data cleaning")
        
        # 删除重复行
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # 处理缺失值
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # 数值列用中位数填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # 分类列用众数填充
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        logger.info("Data cleaning completed")
        return df
    
    def feature_engineering(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 输入数据
            is_training: 是否为训练阶段
            
        Returns:
            特征工程后的数据
        """
        logger.info("Starting feature engineering")
        
        # 数值特征标准化
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.data_config['target_column']]
        
        if numeric_cols:
            if is_training:
                self.scalers['numeric'] = StandardScaler()
                df[numeric_cols] = self.scalers['numeric'].fit_transform(df[numeric_cols])
            else:
                if 'numeric' in self.scalers:
                    df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])
        
        # 分类特征编码
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.data_config['target_column']]
        
        for col in categorical_cols:
            if is_training:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    # 处理未见过的类别
                    unique_values = set(df[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        logger.warning(f"Unknown categories in {col}: {unknown_values}")
                        # 将未知类别替换为最常见的类别
                        most_common = self.encoders[col].classes_[0]
                        df[col] = df[col].astype(str).replace(list(unknown_values), most_common)
                    
                    df[col] = self.encoders[col].transform(df[col].astype(str))
        
        logger.info("Feature engineering completed")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        数据分割
        
        Args:
            df: 输入数据
            
        Returns:
            训练集、验证集、测试集
        """
        logger.info("Splitting data")
        
        target_col = self.data_config['target_column']
        
        # 首先分离出测试集
        train_val, test = train_test_split(
            df, 
            test_size=self.data_config['test_split'],
            stratify=df[target_col] if target_col in df.columns else None,
            random_state=42
        )
        
        # 再从训练验证集中分离出验证集
        val_size = self.data_config['validation_split'] / (1 - self.data_config['test_split'])
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val[target_col] if target_col in train_val.columns else None,
            random_state=42
        )
        
        logger.info(f"Data split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        return train, val, test
    
    def process_pipeline(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        完整的数据处理流水线
        
        Args:
            input_path: 输入数据路径
            output_dir: 输出目录
            
        Returns:
            处理后数据文件路径字典
        """
        logger.info("Starting data processing pipeline")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载和清洗数据
        df = self.load_data(input_path)
        df = self.clean_data(df)
        
        # 特征工程
        df = self.feature_engineering(df, is_training=True)
        
        # 数据分割
        train, val, test = self.split_data(df)
        
        # 保存处理后的数据
        file_paths = {}
        for name, data in [('train', train), ('val', val), ('test', test)]:
            file_path = output_path / f"{name}.parquet"
            data.to_parquet(file_path, index=False)
            file_paths[name] = str(file_path)
            logger.info(f"Saved {name} data to {file_path}")
        
        logger.info("Data processing pipeline completed")
        return file_paths

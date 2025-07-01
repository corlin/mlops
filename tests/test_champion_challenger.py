#!/usr/bin/env python3
"""
冠军挑战者系统测试
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataProcessor
from src.training import LudwigTrainer
from src.tracking import MLflowManager
from src.lifecycle import ChampionChallengerManager


@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    n_samples = 1000
    
    # 生成特征
    X = np.random.randn(n_samples, 5)
    
    # 生成目标变量
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    return df


@pytest.fixture
def temp_config():
    """创建临时配置文件"""
    config = {
        'data': {
            'input_path': 'data/raw/',
            'processed_path': 'data/processed/',
            'validation_split': 0.2,
            'test_split': 0.1,
            'target_column': 'target'
        },
        'ludwig': {
            'config_path': 'config/ludwig_config.yaml',
            'output_directory': 'models/ludwig_output/',
            'experiment_name': 'test_experiment'
        },
        'mlflow': {
            'tracking_uri': 'sqlite:///test_mlflow.db',
            'experiment_name': 'test_champion_challenger',
            'model_registry_name': 'test_models'
        },
        'champion_challenger': {
            'evaluation_metrics': ['accuracy', 'precision', 'recall'],
            'champion_threshold': 0.05,
            'challenger_evaluation_period': 1,
            'shadow_mode_duration': 1,
            'auto_promotion': True
        },
        'deployment': {
            'champion_endpoint': 'http://localhost:8000',
            'challenger_endpoint': 'http://localhost:8001',
            'shadow_mode': True
        },
        'monitoring': {
            'metrics_collection_interval': 60,
            'alert_thresholds': {
                'accuracy_drop': 0.05,
                'latency_increase': 2.0,
                'error_rate': 0.01
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def ludwig_config():
    """创建Ludwig配置文件"""
    config = {
        'model_type': 'ecd',
        'input_features': [
            {
                'name': 'feature_0',
                'type': 'numerical'
            },
            {
                'name': 'feature_1', 
                'type': 'numerical'
            },
            {
                'name': 'category',
                'type': 'category'
            }
        ],
        'output_features': [
            {
                'name': 'target',
                'type': 'binary'
            }
        ],
        'trainer': {
            'epochs': 2,  # 快速测试
            'batch_size': 32
        }
    }
    
    # 确保配置目录存在
    Path('config').mkdir(exist_ok=True)
    
    with open('config/ludwig_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return 'config/ludwig_config.yaml'


class TestDataProcessor:
    """测试数据处理器"""
    
    def test_load_data(self, temp_config, sample_data):
        """测试数据加载"""
        processor = DataProcessor(temp_config)
        
        # 保存测试数据
        test_file = 'test_data.csv'
        sample_data.to_csv(test_file, index=False)
        
        try:
            loaded_data = processor.load_data(test_file)
            assert loaded_data.shape == sample_data.shape
            assert list(loaded_data.columns) == list(sample_data.columns)
        finally:
            Path(test_file).unlink(missing_ok=True)
    
    def test_clean_data(self, temp_config, sample_data):
        """测试数据清洗"""
        processor = DataProcessor(temp_config)
        
        # 添加一些缺失值和重复行
        dirty_data = sample_data.copy()
        dirty_data.loc[0:10, 'feature_0'] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]], ignore_index=True)
        
        cleaned_data = processor.clean_data(dirty_data)
        
        # 检查没有缺失值
        assert cleaned_data.isnull().sum().sum() == 0
        
        # 检查没有重复行
        assert len(cleaned_data) == len(cleaned_data.drop_duplicates())
    
    def test_feature_engineering(self, temp_config, sample_data):
        """测试特征工程"""
        processor = DataProcessor(temp_config)
        
        processed_data = processor.feature_engineering(sample_data, is_training=True)
        
        # 检查数值特征被标准化
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        
        for col in numeric_cols:
            # 标准化后均值应该接近0，标准差接近1
            assert abs(processed_data[col].mean()) < 0.1
            assert abs(processed_data[col].std() - 1.0) < 0.1
    
    def test_split_data(self, temp_config, sample_data):
        """测试数据分割"""
        processor = DataProcessor(temp_config)
        
        train, val, test = processor.split_data(sample_data)
        
        # 检查分割比例
        total_samples = len(sample_data)
        assert len(train) + len(val) + len(test) == total_samples
        
        # 检查没有重叠
        train_indices = set(train.index)
        val_indices = set(val.index)
        test_indices = set(test.index)
        
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0


class TestMLflowManager:
    """测试MLflow管理器"""
    
    def test_init_mlflow_manager(self, temp_config):
        """测试MLflow管理器初始化"""
        manager = MLflowManager(temp_config)
        assert manager.client is not None
        assert manager.experiment_name == 'test_champion_challenger'
    
    def test_log_metrics(self, temp_config):
        """测试指标记录"""
        manager = MLflowManager(temp_config)
        
        # 这里需要实际的MLflow运行来测试
        # 在实际测试中，可能需要mock MLflow
        pass


class TestChampionChallengerManager:
    """测试冠军挑战者管理器"""
    
    def test_init_manager(self, temp_config, ludwig_config):
        """测试管理器初始化"""
        manager = ChampionChallengerManager(temp_config)
        
        assert manager.data_processor is not None
        assert manager.ludwig_trainer is not None
        assert manager.mlflow_manager is not None
    
    def test_state_management(self, temp_config, ludwig_config):
        """测试状态管理"""
        manager = ChampionChallengerManager(temp_config)
        
        # 测试初始状态
        initial_state = manager.get_status()
        assert 'champion_model' in initial_state
        assert 'challenger_models' in initial_state
        assert 'active_shadow_tests' in initial_state
        
        # 测试状态保存和加载
        manager.state['test_key'] = 'test_value'
        manager._save_state()
        
        # 创建新实例并检查状态是否持久化
        new_manager = ChampionChallengerManager(temp_config)
        assert new_manager.state.get('test_key') == 'test_value'


def test_integration_data_processing(temp_config, sample_data):
    """集成测试：数据处理流水线"""
    processor = DataProcessor(temp_config)
    
    # 保存测试数据
    input_file = 'test_input.csv'
    sample_data.to_csv(input_file, index=False)
    
    try:
        # 运行完整的数据处理流水线
        output_dir = 'test_output'
        file_paths = processor.process_pipeline(input_file, output_dir)
        
        # 检查输出文件
        assert 'train' in file_paths
        assert 'val' in file_paths
        assert 'test' in file_paths
        
        for split, path in file_paths.items():
            assert Path(path).exists()
            df = pd.read_parquet(path)
            assert len(df) > 0
            assert 'target' in df.columns
        
    finally:
        # 清理测试文件
        Path(input_file).unlink(missing_ok=True)
        if Path('test_output').exists():
            shutil.rmtree('test_output')


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, '-v'])

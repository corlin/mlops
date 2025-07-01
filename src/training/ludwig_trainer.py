"""
Ludwig训练模块 - 负责使用Ludwig进行模型训练和挑战者生成
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
import mlflow
try:
    import mlflow.ludwig
    MLFLOW_LUDWIG_AVAILABLE = True
except ImportError:
    MLFLOW_LUDWIG_AVAILABLE = False
    logger.warning("mlflow.ludwig not available, will use generic MLflow logging")
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt
import tempfile
import shutil


class LudwigTrainer:
    """Ludwig训练器，负责训练挑战者模型"""

    def _log_ludwig_model(self, model, artifact_path: str, registered_model_name: str = None):
        """记录Ludwig模型到MLflow"""
        try:
            if MLFLOW_LUDWIG_AVAILABLE:
                mlflow.ludwig.log_model(
                    ludwig_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            else:
                # 使用通用方法记录模型
                import pickle
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, "model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)

                    mlflow.log_artifact(model_path, artifact_path)

                    if registered_model_name:
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                        mlflow.register_model(model_uri, registered_model_name)

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            # 至少记录一些基本信息
            mlflow.log_param("model_type", "ludwig")
            mlflow.log_param("model_logged", "failed")

    def __init__(self, config_path: str):
        """
        初始化Ludwig训练器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.ludwig_config_path = self.config['ludwig']['config_path']
        self.output_directory = self.config['ludwig']['output_directory']
        self.experiment_name = self.config['ludwig']['experiment_name']
        
        # 加载Ludwig配置
        with open(self.ludwig_config_path, 'r', encoding='utf-8') as f:
            self.ludwig_config = yaml.safe_load(f)
            
        # 创建输出目录
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self):
        """设置MLflow配置"""
        try:
            # 设置tracking URI
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
            mlflow.set_tracking_uri(tracking_uri)

            # 确保本地artifact目录存在
            project_root = Path(__file__).parent.parent.parent
            artifact_dir = project_root / "mlflow_artifacts"
            artifact_dir.mkdir(exist_ok=True)

            # 设置artifact URI（如果使用本地存储）
            if tracking_uri.startswith('file://') or 'localhost' in tracking_uri:
                os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = str(artifact_dir)

            logger.info(f"MLflow tracking URI: {tracking_uri}")
            logger.info(f"MLflow artifact directory: {artifact_dir}")

        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            # 使用默认设置
            pass

    def _setup_experiment(self):
        """设置MLflow实验，处理删除的实验"""
        try:
            # 尝试获取现有实验
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment and experiment.lifecycle_stage == 'deleted':
                # 如果实验被删除，创建新的实验名称
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_experiment_name = f"{self.experiment_name}_{timestamp}"

                logger.warning(f"Experiment '{self.experiment_name}' was deleted")
                logger.info(f"Creating new experiment: {new_experiment_name}")

                # 创建新实验
                experiment_id = mlflow.create_experiment(new_experiment_name)
                self.experiment_name = new_experiment_name

            elif experiment:
                # 实验存在且未删除
                experiment_id = experiment.experiment_id

            else:
                # 实验不存在，创建新的
                logger.info(f"Creating new experiment: {self.experiment_name}")
                experiment_id = mlflow.create_experiment(self.experiment_name)

            # 设置实验
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Using MLflow experiment: {self.experiment_name} (ID: {experiment_id})")

        except Exception as e:
            logger.warning(f"Failed to setup experiment: {e}")
            # 使用默认实验
            logger.info("Falling back to default experiment")
            mlflow.set_experiment("Default")
        
    def prepare_ludwig_data(self, train_path: str, val_path: str, test_path: str) -> Dict[str, str]:
        """
        准备Ludwig训练数据
        
        Args:
            train_path: 训练数据路径
            val_path: 验证数据路径
            test_path: 测试数据路径
            
        Returns:
            数据路径字典
        """
        logger.info("Preparing Ludwig training data")
        
        # Ludwig支持直接使用parquet文件
        data_paths = {
            'train': train_path,
            'val': val_path,  # 使用'val'保持与数据处理器一致
            'test': test_path
        }
        
        # 验证数据文件存在
        for split, path in data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{split} data file not found: {path}")
                
        logger.info("Ludwig data preparation completed")
        return data_paths
    
    def update_ludwig_config(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据数据信息更新Ludwig配置
        
        Args:
            data_info: 数据信息
            
        Returns:
            更新后的Ludwig配置
        """
        logger.info("Updating Ludwig configuration")
        
        config = self.ludwig_config.copy()
        
        # 如果有特征列信息，更新配置
        if 'feature_columns' in data_info:
            # 这里可以根据实际数据动态调整特征配置
            pass
            
        return config
    
    def train_model(self, data_paths: Dict[str, str], model_name: str = "challenger") -> Dict[str, Any]:
        """
        训练Ludwig模型
        
        Args:
            data_paths: 数据路径字典
            model_name: 模型名称
            
        Returns:
            训练结果
        """
        logger.info(f"Starting Ludwig model training: {model_name}")

        # 设置MLflow tracking URI和artifact路径
        self._setup_mlflow()

        # 设置MLflow实验（处理删除的实验）
        self._setup_experiment()
        
        with mlflow.start_run(run_name=f"{model_name}_training") as run:
            try:
                # 创建临时输出目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 初始化Ludwig模型
                    model = LudwigModel(
                        config=self.ludwig_config,
                        logging_level='INFO'
                    )
                    
                    # 训练模型
                    train_stats, preprocessed_data, output_directory = model.train(
                        dataset=data_paths['train'],
                        validation_set=data_paths['val'],  # 修复键名
                        test_set=data_paths['test'],
                        output_directory=temp_dir
                    )
                    
                    # 记录训练指标
                    if train_stats:
                        try:
                            # Ludwig TrainingStats对象的正确访问方式
                            if hasattr(train_stats, 'training'):
                                training_metrics = train_stats.training
                                if isinstance(training_metrics, dict):
                                    for feature_name, feature_metrics in training_metrics.items():
                                        if isinstance(feature_metrics, dict):
                                            for metric_name, metric_values in feature_metrics.items():
                                                if isinstance(metric_values, list) and metric_values:
                                                    # 记录最后一个epoch的值
                                                    final_value = metric_values[-1]
                                                    if isinstance(final_value, (int, float)):
                                                        mlflow.log_metric(f"train_{feature_name}_{metric_name}", final_value)
                        except Exception as e:
                            logger.warning(f"Failed to log training metrics: {e}")
                            # 尝试直接转换为字典
                            try:
                                if hasattr(train_stats, '__dict__'):
                                    stats_dict = train_stats.__dict__
                                    for key, value in stats_dict.items():
                                        if isinstance(value, (int, float)):
                                            mlflow.log_metric(f"train_{key}", value)
                            except Exception:
                                logger.warning("Could not extract training metrics")
                    
                    # 评估模型
                    try:
                        # Ludwig evaluate()方法的返回值在不同版本中可能不同
                        eval_results = model.evaluate(
                            dataset=data_paths['test'],
                            collect_predictions=True,
                            collect_overall_stats=True
                        )

                        # 处理不同的返回值格式
                        if isinstance(eval_results, tuple):
                            if len(eval_results) == 2:
                                test_results, predictions = eval_results
                            elif len(eval_results) == 3:
                                test_results, predictions, _ = eval_results
                            elif len(eval_results) == 1:
                                test_results = eval_results[0]
                                predictions = None
                            else:
                                # 处理其他情况
                                test_results = eval_results[0] if eval_results else None
                                predictions = eval_results[1] if len(eval_results) > 1 else None
                        else:
                            # 如果不是tuple，直接使用
                            test_results = eval_results
                            predictions = None

                    except Exception as e:
                        logger.warning(f"Model evaluation failed: {e}")
                        test_results = None
                        predictions = None
                    
                    # 记录测试指标
                    if test_results:
                        try:
                            # 处理Ludwig评估结果
                            if isinstance(test_results, dict):
                                for feature_name, feature_metrics in test_results.items():
                                    if isinstance(feature_metrics, dict):
                                        for metric_name, metric_value in feature_metrics.items():
                                            if isinstance(metric_value, (int, float)):
                                                mlflow.log_metric(f"test_{feature_name}_{metric_name}", metric_value)
                            else:
                                # 尝试访问对象属性
                                if hasattr(test_results, '__dict__'):
                                    for key, value in test_results.__dict__.items():
                                        if isinstance(value, (int, float)):
                                            mlflow.log_metric(f"test_{key}", value)
                        except Exception as e:
                            logger.warning(f"Failed to log test metrics: {e}")
                    
                    # 保存模型
                    model_output_path = Path(self.output_directory) / f"{model_name}_{run.info.run_id}"
                    model_output_path.mkdir(parents=True, exist_ok=True)
                    
                    # 复制模型文件
                    if os.path.exists(output_directory):
                        shutil.copytree(output_directory, model_output_path, dirs_exist_ok=True)
                    
                    # 记录模型到MLflow
                    self._log_ludwig_model(
                        model=model,
                        artifact_path="model",
                        registered_model_name=f"{model_name}_model"
                    )
                    
                    # 记录配置（避免写入artifact）
                    try:
                        # 记录配置参数而不是文件
                        mlflow.log_param("model_type", self.ludwig_config.get('model_type', 'ecd'))
                        mlflow.log_param("num_input_features", len(self.ludwig_config.get('input_features', [])))
                        mlflow.log_param("num_output_features", len(self.ludwig_config.get('output_features', [])))

                        # 记录训练器配置
                        trainer_config = self.ludwig_config.get('trainer', {})
                        for key, value in trainer_config.items():
                            if isinstance(value, (int, float, str, bool)):
                                mlflow.log_param(f"trainer_{key}", value)

                        logger.info("Ludwig configuration logged as parameters")
                    except Exception as e:
                        logger.warning(f"Failed to log configuration: {e}")
                    
                    training_result = {
                        'run_id': run.info.run_id,
                        'model_path': str(model_output_path),
                        'train_stats': train_stats,
                        'test_results': test_results,
                        'model': model
                    }
                    
                    logger.info(f"Model training completed: {model_name}")
                    return training_result
                    
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def hyperparameter_optimization(self, data_paths: Dict[str, str], model_name: str = "challenger_hpo") -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            data_paths: 数据路径字典
            model_name: 模型名称
            
        Returns:
            优化结果
        """
        logger.info(f"Starting hyperparameter optimization: {model_name}")

        # 检查Ray是否可用
        try:
            from ray.air import Checkpoint
            ray_available = True
        except ImportError as e:
            logger.warning(f"Ray not available for hyperparameter optimization: {e}")
            logger.info("Falling back to default configuration")
            ray_available = False

        if not ray_available:
            # 返回默认配置，跳过超参数优化
            logger.info("Skipping hyperparameter optimization due to Ray unavailability")
            return {
                'best_config': self.ludwig_config.copy(),
                'best_score': None,
                'optimization_skipped': True,
                'reason': 'Ray not available'
            }

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_hpo") as run:
            try:
                # 创建临时输出目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 执行超参数优化
                    hyperopt_results = hyperopt(
                        config=self.ludwig_config,
                        dataset=data_paths['train'],
                        validation_set=data_paths['val'],  # 修复键名：'val' 而不是 'validation'
                        test_set=data_paths['test'],
                        output_directory=temp_dir
                    )
                    
                    # 获取最佳配置
                    best_config = hyperopt_results.best_trial_parameters
                    best_score = hyperopt_results.best_trial_stats
                    
                    # 记录最佳参数
                    mlflow.log_params(best_config)
                    mlflow.log_metrics(best_score)
                    
                    # 使用最佳配置训练最终模型
                    final_config = self.ludwig_config.copy()
                    final_config.update(best_config)
                    
                    model = LudwigModel(config=final_config)
                    train_stats, _, output_directory = model.train(
                        dataset=data_paths['train'],
                        validation_set=data_paths['val'],  # 修复键名
                        test_set=data_paths['test'],
                        output_directory=temp_dir
                    )
                    
                    # 保存优化后的模型
                    model_output_path = Path(self.output_directory) / f"{model_name}_{run.info.run_id}"
                    model_output_path.mkdir(parents=True, exist_ok=True)
                    
                    if os.path.exists(output_directory):
                        shutil.copytree(output_directory, model_output_path, dirs_exist_ok=True)
                    
                    # 记录模型
                    self._log_ludwig_model(
                        model=model,
                        artifact_path="model",
                        registered_model_name=f"{model_name}_optimized"
                    )
                    
                    hpo_result = {
                        'run_id': run.info.run_id,
                        'model_path': str(model_output_path),
                        'best_config': best_config,
                        'best_score': best_score,
                        'hyperopt_results': hyperopt_results,
                        'model': model
                    }
                    
                    logger.info(f"Hyperparameter optimization completed: {model_name}")
                    return hpo_result
                    
            except Exception as e:
                logger.error(f"HPO failed for {model_name}: {str(e)}")
                mlflow.log_param("hpo_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def generate_challenger(self, data_paths: Dict[str, str], use_hpo: bool = True) -> Dict[str, Any]:
        """
        生成挑战者模型
        
        Args:
            data_paths: 数据路径字典
            use_hpo: 是否使用超参数优化
            
        Returns:
            挑战者模型信息
        """
        logger.info("Generating challenger model")

        if use_hpo:
            hpo_result = self.hyperparameter_optimization(data_paths, "challenger_hpo")

            # 检查是否跳过了超参数优化
            if hpo_result.get('optimization_skipped', False):
                logger.info("HPO was skipped, falling back to standard training")
                result = self.train_model(data_paths, "challenger")
            else:
                result = hpo_result
        else:
            result = self.train_model(data_paths, "challenger")

        logger.info("Challenger model generation completed")
        return result

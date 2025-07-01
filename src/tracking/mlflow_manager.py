"""
MLflow管理模块 - 负责实验跟踪、模型注册和版本管理
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
try:
    import mlflow.ludwig
    MLFLOW_LUDWIG_AVAILABLE = True
except ImportError:
    MLFLOW_LUDWIG_AVAILABLE = False
from mlflow.entities import ViewType
import pandas as pd
import yaml
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta
import json


class MLflowManager:
    """MLflow管理器，负责实验跟踪和模型注册"""
    
    def __init__(self, config_path: str):
        """
        初始化MLflow管理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_config = self.config['mlflow']
        self.champion_challenger_config = self.config['champion_challenger']
        
        # 设置MLflow跟踪URI
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        
        # 初始化MLflow客户端
        self.client = MlflowClient()
        
        # 设置实验
        self.experiment_name = self.mlflow_config['experiment_name']
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.mlflow_config.get('artifact_location')
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def log_model_metrics(self, run_id: str, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        记录模型指标
        
        Args:
            run_id: MLflow运行ID
            metrics: 指标字典
            prefix: 指标前缀
        """
        logger.info(f"Logging metrics for run {run_id}")
        
        with mlflow.start_run(run_id=run_id):
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    full_metric_name = f"{prefix}_{metric_name}" if prefix else metric_name
                    mlflow.log_metric(full_metric_name, metric_value)
    
    def register_model(self, run_id: str, model_name: str, model_version_tags: Dict[str, str] = None) -> str:
        """
        注册模型到模型注册表
        
        Args:
            run_id: MLflow运行ID
            model_name: 模型名称
            model_version_tags: 模型版本标签
            
        Returns:
            模型版本号
        """
        logger.info(f"Registering model {model_name} from run {run_id}")
        
        try:
            # 获取模型URI
            model_uri = f"runs:/{run_id}/model"
            
            # 注册模型
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=model_version_tags
            )
            
            logger.info(f"Model registered: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def get_model_versions(self, model_name: str, stages: List[str] = None) -> List[Dict[str, Any]]:
        """
        获取模型版本信息
        
        Args:
            model_name: 模型名称
            stages: 模型阶段列表
            
        Returns:
            模型版本信息列表
        """
        logger.info(f"Getting model versions for {model_name}")
        
        try:
            model_versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if stages:
                model_versions = [mv for mv in model_versions if mv.current_stage in stages]
            
            versions_info = []
            for mv in model_versions:
                version_info = {
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'creation_timestamp': mv.creation_timestamp,
                    'last_updated_timestamp': mv.last_updated_timestamp,
                    'run_id': mv.run_id,
                    'tags': mv.tags
                }
                versions_info.append(version_info)
            
            return versions_info
            
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []
    
    def transition_model_stage(self, model_name: str, version: str, stage: str, archive_existing: bool = True) -> None:
        """
        转换模型阶段
        
        Args:
            model_name: 模型名称
            version: 模型版本
            stage: 目标阶段
            archive_existing: 是否归档现有模型
        """
        logger.info(f"Transitioning model {model_name} v{version} to {stage}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def compare_models(self, champion_run_id: str, challenger_run_id: str) -> Dict[str, Any]:
        """
        比较冠军和挑战者模型
        
        Args:
            champion_run_id: 冠军模型运行ID
            challenger_run_id: 挑战者模型运行ID
            
        Returns:
            比较结果
        """
        logger.info(f"Comparing champion {champion_run_id} vs challenger {challenger_run_id}")
        
        try:
            # 获取运行信息
            champion_run = self.client.get_run(champion_run_id)
            challenger_run = self.client.get_run(challenger_run_id)
            
            # 提取指标
            champion_metrics = champion_run.data.metrics
            challenger_metrics = challenger_run.data.metrics
            
            # 比较指标
            comparison_result = {
                'champion_run_id': champion_run_id,
                'challenger_run_id': challenger_run_id,
                'champion_metrics': champion_metrics,
                'challenger_metrics': challenger_metrics,
                'improvements': {},
                'recommendation': 'keep_champion'
            }
            
            # 计算改进
            evaluation_metrics = self.champion_challenger_config['evaluation_metrics']
            threshold = self.champion_challenger_config['champion_threshold']
            
            significant_improvements = 0
            total_improvements = 0
            
            for metric in evaluation_metrics:
                if metric in champion_metrics and metric in challenger_metrics:
                    champion_value = champion_metrics[metric]
                    challenger_value = challenger_metrics[metric]
                    
                    improvement = (challenger_value - champion_value) / champion_value
                    comparison_result['improvements'][metric] = {
                        'champion': champion_value,
                        'challenger': challenger_value,
                        'improvement': improvement,
                        'significant': improvement > threshold
                    }
                    
                    if improvement > 0:
                        total_improvements += 1
                        if improvement > threshold:
                            significant_improvements += 1
            
            # 决策逻辑
            if significant_improvements > 0 and significant_improvements >= len(evaluation_metrics) * 0.5:
                comparison_result['recommendation'] = 'promote_challenger'
            elif total_improvements > len(evaluation_metrics) * 0.7:
                comparison_result['recommendation'] = 'shadow_test'
            
            logger.info(f"Model comparison completed: {comparison_result['recommendation']}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def get_champion_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取当前冠军模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            冠军模型信息
        """
        logger.info(f"Getting champion model for {model_name}")
        
        try:
            champion_versions = self.get_model_versions(model_name, stages=['Production'])
            
            if champion_versions:
                # 返回最新的生产模型
                champion = max(champion_versions, key=lambda x: x['last_updated_timestamp'])
                return champion
            else:
                logger.warning(f"No champion model found for {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get champion model: {e}")
            return None
    
    def get_experiment_runs(self, experiment_name: str = None, max_results: int = 100) -> pd.DataFrame:
        """
        获取实验运行记录
        
        Args:
            experiment_name: 实验名称
            max_results: 最大结果数
            
        Returns:
            运行记录DataFrame
        """
        exp_name = experiment_name or self.experiment_name
        logger.info(f"Getting experiment runs for {exp_name}")
        
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                logger.warning(f"Experiment {exp_name} not found")
                return pd.DataFrame()
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results
            )
            
            return runs
            
        except Exception as e:
            logger.error(f"Failed to get experiment runs: {e}")
            return pd.DataFrame()
    
    def cleanup_old_runs(self, days_to_keep: int = 30) -> None:
        """
        清理旧的实验运行
        
        Args:
            days_to_keep: 保留天数
        """
        logger.info(f"Cleaning up runs older than {days_to_keep} days")
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"attribute.start_time < {cutoff_timestamp}",
                run_view_type=ViewType.ACTIVE_ONLY
            )
            
            deleted_count = 0
            for run in runs:
                # 只删除非生产模型的运行
                if not any(tag.key == 'stage' and tag.value == 'Production' 
                          for tag in run.data.tags.items()):
                    self.client.delete_run(run.info.run_id)
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} old runs")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            raise

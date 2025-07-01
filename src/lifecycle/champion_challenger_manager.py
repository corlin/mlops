"""
冠军挑战者生命周期管理模块 - 核心业务逻辑
"""

import yaml
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

from ..data import DataProcessor
from ..training import LudwigTrainer
from ..tracking import MLflowManager
from ..deployment import ModelDeployer
from ..monitoring import ModelMonitor


class ChampionChallengerManager:
    """冠军挑战者生命周期管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化管理器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.cc_config = self.config['champion_challenger']
        
        # 初始化各个组件
        self.data_processor = DataProcessor(config_path)
        self.ludwig_trainer = LudwigTrainer(config_path)
        self.mlflow_manager = MLflowManager(config_path)
        self.model_deployer = ModelDeployer(config_path)
        self.model_monitor = ModelMonitor(config_path)
        
        # 状态管理
        self.state_file = Path("state/champion_challenger_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'champion_model': None,
            'challenger_models': [],
            'shadow_tests': [],
            'last_evaluation': None
        }
    
    def _save_state(self) -> None:
        """保存状态"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def train_challenger(self, data_path: str, model_name: str = None) -> Dict[str, Any]:
        """
        训练新的挑战者模型
        
        Args:
            data_path: 训练数据路径
            model_name: 模型名称
            
        Returns:
            训练结果
        """
        logger.info("Starting challenger training process")
        
        # 生成模型名称
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"challenger_{timestamp}"
        
        try:
            # 1. 数据处理
            logger.info("Processing training data")
            processed_data_dir = f"data/processed/{model_name}"
            data_paths = self.data_processor.process_pipeline(data_path, processed_data_dir)
            
            # 2. 训练模型
            logger.info("Training challenger model")
            training_result = self.ludwig_trainer.generate_challenger(data_paths, use_hpo=True)
            
            # 3. 注册模型
            logger.info("Registering challenger model")
            model_version = self.mlflow_manager.register_model(
                run_id=training_result['run_id'],
                model_name=model_name,
                model_version_tags={
                    'type': 'challenger',
                    'training_date': datetime.now().isoformat(),
                    'data_path': data_path
                }
            )
            
            # 4. 更新状态
            challenger_info = {
                'name': model_name,
                'version': model_version,
                'run_id': training_result['run_id'],
                'training_date': datetime.now().isoformat(),
                'status': 'trained',
                'metrics': training_result.get('test_results', {})
            }
            
            self.state['challenger_models'].append(challenger_info)
            self._save_state()
            
            logger.info(f"Challenger training completed: {model_name}")
            return challenger_info
            
        except Exception as e:
            logger.error(f"Challenger training failed: {e}")
            raise
    
    def evaluate_challenger(self, challenger_name: str) -> Dict[str, Any]:
        """
        评估挑战者模型与冠军模型
        
        Args:
            challenger_name: 挑战者模型名称
            
        Returns:
            评估结果
        """
        logger.info(f"Evaluating challenger: {challenger_name}")
        
        try:
            # 获取挑战者信息
            challenger = None
            for c in self.state['challenger_models']:
                if c['name'] == challenger_name:
                    challenger = c
                    break
            
            if not challenger:
                raise ValueError(f"Challenger {challenger_name} not found")
            
            # 获取当前冠军
            champion = self.state.get('champion_model')
            if not champion:
                logger.info("No champion model found, promoting challenger directly")
                return self._promote_challenger(challenger_name)
            
            # 比较模型
            comparison_result = self.mlflow_manager.compare_models(
                champion_run_id=champion['run_id'],
                challenger_run_id=challenger['run_id']
            )
            
            # 根据比较结果决定下一步
            recommendation = comparison_result['recommendation']
            
            if recommendation == 'promote_challenger':
                logger.info("Challenger shows significant improvement, promoting to champion")
                return self._promote_challenger(challenger_name)
            elif recommendation == 'shadow_test':
                logger.info("Challenger shows improvement, starting shadow test")
                return self._start_shadow_test(challenger_name)
            else:
                logger.info("Challenger does not show sufficient improvement")
                return {
                    'action': 'keep_champion',
                    'comparison': comparison_result,
                    'challenger': challenger_name
                }
                
        except Exception as e:
            logger.error(f"Challenger evaluation failed: {e}")
            raise
    
    def _promote_challenger(self, challenger_name: str) -> Dict[str, Any]:
        """
        提升挑战者为冠军
        
        Args:
            challenger_name: 挑战者名称
            
        Returns:
            提升结果
        """
        logger.info(f"Promoting challenger to champion: {challenger_name}")
        
        try:
            # 找到挑战者
            challenger = None
            for i, c in enumerate(self.state['challenger_models']):
                if c['name'] == challenger_name:
                    challenger = c
                    challenger_index = i
                    break
            
            if not challenger:
                raise ValueError(f"Challenger {challenger_name} not found")
            
            # 如果有现有冠军，先归档
            if self.state.get('champion_model'):
                old_champion = self.state['champion_model']
                logger.info(f"Archiving old champion: {old_champion['name']}")
                
                # 转换旧冠军到归档状态
                self.mlflow_manager.transition_model_stage(
                    model_name=old_champion['name'],
                    version=old_champion['version'],
                    stage='Archived'
                )
            
            # 提升挑战者到生产环境
            self.mlflow_manager.transition_model_stage(
                model_name=challenger['name'],
                version=challenger['version'],
                stage='Production'
            )
            
            # 部署新冠军
            deployment_result = self.model_deployer.deploy_champion(
                model_name=challenger['name'],
                model_version=challenger['version']
            )
            
            # 更新状态
            challenger['status'] = 'champion'
            challenger['promotion_date'] = datetime.now().isoformat()
            challenger['deployment_info'] = deployment_result
            
            self.state['champion_model'] = challenger
            self.state['challenger_models'].pop(challenger_index)
            self._save_state()
            
            logger.info(f"Champion promotion completed: {challenger_name}")
            return {
                'action': 'promoted_to_champion',
                'new_champion': challenger,
                'deployment': deployment_result
            }
            
        except Exception as e:
            logger.error(f"Champion promotion failed: {e}")
            raise
    
    def _start_shadow_test(self, challenger_name: str) -> Dict[str, Any]:
        """
        开始影子测试
        
        Args:
            challenger_name: 挑战者名称
            
        Returns:
            影子测试结果
        """
        logger.info(f"Starting shadow test for: {challenger_name}")
        
        try:
            # 找到挑战者
            challenger = None
            for c in self.state['challenger_models']:
                if c['name'] == challenger_name:
                    challenger = c
                    break
            
            if not challenger:
                raise ValueError(f"Challenger {challenger_name} not found")
            
            # 部署挑战者到影子环境
            shadow_deployment = self.model_deployer.deploy_shadow(
                model_name=challenger['name'],
                model_version=challenger['version']
            )
            
            # 创建影子测试记录
            shadow_test = {
                'challenger_name': challenger_name,
                'start_date': datetime.now().isoformat(),
                'end_date': (datetime.now() + timedelta(days=self.cc_config['shadow_mode_duration'])).isoformat(),
                'status': 'running',
                'deployment_info': shadow_deployment,
                'metrics_collected': []
            }
            
            # 更新状态
            challenger['status'] = 'shadow_testing'
            self.state['shadow_tests'].append(shadow_test)
            self._save_state()
            
            logger.info(f"Shadow test started: {challenger_name}")
            return {
                'action': 'shadow_test_started',
                'shadow_test': shadow_test,
                'challenger': challenger
            }
            
        except Exception as e:
            logger.error(f"Shadow test start failed: {e}")
            raise
    
    def monitor_shadow_tests(self) -> List[Dict[str, Any]]:
        """
        监控影子测试
        
        Returns:
            监控结果列表
        """
        logger.info("Monitoring shadow tests")
        
        results = []
        current_time = datetime.now()
        
        for shadow_test in self.state['shadow_tests']:
            if shadow_test['status'] != 'running':
                continue
            
            try:
                # 检查是否到期
                end_date = datetime.fromisoformat(shadow_test['end_date'])
                if current_time >= end_date:
                    logger.info(f"Shadow test completed: {shadow_test['challenger_name']}")
                    result = self._complete_shadow_test(shadow_test)
                    results.append(result)
                else:
                    # 收集监控指标
                    metrics = self.model_monitor.collect_shadow_metrics(
                        shadow_test['challenger_name']
                    )
                    shadow_test['metrics_collected'].append({
                        'timestamp': current_time.isoformat(),
                        'metrics': metrics
                    })
                    results.append({
                        'challenger_name': shadow_test['challenger_name'],
                        'status': 'monitoring',
                        'metrics': metrics
                    })
                    
            except Exception as e:
                logger.error(f"Shadow test monitoring failed for {shadow_test['challenger_name']}: {e}")
                shadow_test['status'] = 'failed'
                shadow_test['error'] = str(e)
        
        self._save_state()
        return results
    
    def _complete_shadow_test(self, shadow_test: Dict[str, Any]) -> Dict[str, Any]:
        """
        完成影子测试
        
        Args:
            shadow_test: 影子测试信息
            
        Returns:
            完成结果
        """
        challenger_name = shadow_test['challenger_name']
        logger.info(f"Completing shadow test: {challenger_name}")
        
        try:
            # 分析影子测试结果
            analysis_result = self.model_monitor.analyze_shadow_test_results(shadow_test)
            
            # 决定是否提升
            if analysis_result['recommendation'] == 'promote':
                logger.info(f"Shadow test successful, promoting challenger: {challenger_name}")
                promotion_result = self._promote_challenger(challenger_name)
                shadow_test['status'] = 'completed_promoted'
                return {
                    'action': 'shadow_test_completed_promoted',
                    'challenger_name': challenger_name,
                    'analysis': analysis_result,
                    'promotion': promotion_result
                }
            else:
                logger.info(f"Shadow test not successful, keeping champion: {challenger_name}")
                # 清理影子部署
                self.model_deployer.cleanup_shadow(challenger_name)
                shadow_test['status'] = 'completed_rejected'
                return {
                    'action': 'shadow_test_completed_rejected',
                    'challenger_name': challenger_name,
                    'analysis': analysis_result
                }
                
        except Exception as e:
            logger.error(f"Shadow test completion failed: {e}")
            shadow_test['status'] = 'failed'
            shadow_test['error'] = str(e)
            raise
    
    def run_lifecycle_cycle(self, data_path: str = None) -> Dict[str, Any]:
        """
        运行完整的生命周期循环
        
        Args:
            data_path: 新数据路径（可选）
            
        Returns:
            循环结果
        """
        logger.info("Running champion-challenger lifecycle cycle")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'errors': []
        }
        
        try:
            # 1. 监控现有影子测试
            shadow_results = self.monitor_shadow_tests()
            if shadow_results:
                results['actions_taken'].append({
                    'action': 'shadow_monitoring',
                    'results': shadow_results
                })
            
            # 2. 如果有新数据，训练新挑战者
            if data_path:
                try:
                    challenger_result = self.train_challenger(data_path)
                    results['actions_taken'].append({
                        'action': 'challenger_training',
                        'result': challenger_result
                    })
                    
                    # 3. 评估新挑战者
                    evaluation_result = self.evaluate_challenger(challenger_result['name'])
                    results['actions_taken'].append({
                        'action': 'challenger_evaluation',
                        'result': evaluation_result
                    })
                    
                except Exception as e:
                    logger.error(f"Challenger training/evaluation failed: {e}")
                    results['errors'].append({
                        'action': 'challenger_training_evaluation',
                        'error': str(e)
                    })
            
            # 4. 清理旧的实验运行
            try:
                self.mlflow_manager.cleanup_old_runs(days_to_keep=30)
                results['actions_taken'].append({
                    'action': 'cleanup_old_runs',
                    'result': 'completed'
                })
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                results['errors'].append({
                    'action': 'cleanup',
                    'error': str(e)
                })
            
            # 5. 更新最后评估时间
            self.state['last_evaluation'] = datetime.now().isoformat()
            self._save_state()
            
            logger.info("Lifecycle cycle completed")
            return results
            
        except Exception as e:
            logger.error(f"Lifecycle cycle failed: {e}")
            results['errors'].append({
                'action': 'lifecycle_cycle',
                'error': str(e)
            })
            return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            状态信息
        """
        return {
            'champion_model': self.state.get('champion_model'),
            'challenger_models': self.state.get('challenger_models', []),
            'active_shadow_tests': [st for st in self.state.get('shadow_tests', []) if st['status'] == 'running'],
            'last_evaluation': self.state.get('last_evaluation'),
            'system_health': self.model_monitor.get_system_health()
        }

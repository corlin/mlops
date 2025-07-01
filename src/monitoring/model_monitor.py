"""
模型监控模块 - 负责性能监控、指标收集和自动化决策
"""

import yaml
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta
import json
import time
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import sqlite3
from pathlib import Path


class ModelMonitor:
    """模型监控器，负责性能监控和指标收集"""
    
    def __init__(self, config_path: str):
        """
        初始化监控器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.monitoring_config = self.config['monitoring']
        self.deployment_config = self.config['deployment']
        self.cc_config = self.config['champion_challenger']
        
        # 初始化Prometheus指标
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # 初始化数据库
        self.db_path = Path("monitoring/metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_database()
    
    def _setup_prometheus_metrics(self):
        """设置Prometheus指标"""
        self.metrics = {
            'prediction_count': Counter(
                'model_predictions_total',
                'Total number of predictions made',
                ['model_name', 'model_version', 'model_type'],
                registry=self.registry
            ),
            'prediction_latency': Histogram(
                'model_prediction_latency_seconds',
                'Prediction latency in seconds',
                ['model_name', 'model_version', 'model_type'],
                registry=self.registry
            ),
            'prediction_errors': Counter(
                'model_prediction_errors_total',
                'Total number of prediction errors',
                ['model_name', 'model_version', 'model_type', 'error_type'],
                registry=self.registry
            ),
            'model_accuracy': Gauge(
                'model_accuracy',
                'Current model accuracy',
                ['model_name', 'model_version', 'model_type'],
                registry=self.registry
            ),
            'data_drift_score': Gauge(
                'model_data_drift_score',
                'Data drift detection score',
                ['model_name', 'model_version'],
                registry=self.registry
            )
        }
    
    def _setup_database(self):
        """设置监控数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    input_data TEXT,
                    prediction TEXT,
                    latency REAL,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp 
                ON model_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp 
                ON prediction_logs(timestamp)
            """)
    
    def collect_model_metrics(self, model_endpoint: str, model_info: Dict[str, str]) -> Dict[str, Any]:
        """
        收集模型指标
        
        Args:
            model_endpoint: 模型服务端点
            model_info: 模型信息
            
        Returns:
            收集的指标
        """
        logger.info(f"Collecting metrics from {model_endpoint}")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_info['name'],
            'model_version': model_info['version'],
            'model_type': model_info.get('type', 'unknown'),
            'endpoint': model_endpoint,
            'health_status': 'unknown',
            'response_time': None,
            'error_rate': 0.0,
            'throughput': 0.0
        }
        
        try:
            # 健康检查
            start_time = time.time()
            health_response = requests.get(f"{model_endpoint}/health", timeout=10)
            response_time = time.time() - start_time
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                metrics['health_status'] = health_data.get('status', 'unknown')
                metrics['response_time'] = response_time
            else:
                metrics['health_status'] = 'unhealthy'
                
        except Exception as e:
            logger.error(f"Health check failed for {model_endpoint}: {e}")
            metrics['health_status'] = 'error'
            metrics['error_message'] = str(e)
        
        # 从数据库获取历史指标
        try:
            historical_metrics = self._get_historical_metrics(
                model_info['name'], 
                model_info['version'],
                hours=1
            )
            
            if historical_metrics:
                # 计算错误率
                total_predictions = len(historical_metrics)
                error_predictions = sum(1 for m in historical_metrics if m.get('error_message'))
                metrics['error_rate'] = error_predictions / total_predictions if total_predictions > 0 else 0.0
                
                # 计算吞吐量（每分钟预测数）
                metrics['throughput'] = total_predictions / 60.0  # 假设1小时内的数据
                
        except Exception as e:
            logger.error(f"Failed to calculate historical metrics: {e}")
        
        # 记录指标到数据库
        self._store_metrics(metrics)
        
        # 更新Prometheus指标
        self._update_prometheus_metrics(metrics)
        
        return metrics
    
    def collect_shadow_metrics(self, challenger_name: str) -> Dict[str, Any]:
        """
        收集影子模型指标
        
        Args:
            challenger_name: 挑战者模型名称
            
        Returns:
            影子模型指标
        """
        logger.info(f"Collecting shadow metrics for {challenger_name}")
        
        # 获取影子模型端点
        shadow_endpoint = self.deployment_config['challenger_endpoint']
        
        model_info = {
            'name': challenger_name,
            'version': 'shadow',
            'type': 'challenger'
        }
        
        # 收集基础指标
        shadow_metrics = self.collect_model_metrics(shadow_endpoint, model_info)
        
        # 收集影子特定指标
        try:
            # 比较影子模型和冠军模型的预测结果
            comparison_metrics = self._compare_shadow_predictions(challenger_name)
            shadow_metrics.update(comparison_metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect shadow comparison metrics: {e}")
            shadow_metrics['comparison_error'] = str(e)
        
        return shadow_metrics
    
    def _compare_shadow_predictions(self, challenger_name: str) -> Dict[str, Any]:
        """
        比较影子模型和冠军模型的预测结果
        
        Args:
            challenger_name: 挑战者名称
            
        Returns:
            比较指标
        """
        # 这里实现影子模型和冠军模型预测结果的比较逻辑
        # 在实际应用中，需要从预测日志中获取数据进行比较
        
        comparison_metrics = {
            'prediction_agreement_rate': 0.0,
            'average_prediction_difference': 0.0,
            'confidence_correlation': 0.0
        }
        
        try:
            # 从数据库获取最近的预测日志
            with sqlite3.connect(self.db_path) as conn:
                # 获取冠军模型预测
                champion_query = """
                    SELECT prediction, timestamp FROM prediction_logs 
                    WHERE model_type = 'champion' 
                    AND timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp DESC LIMIT 100
                """
                champion_predictions = pd.read_sql_query(champion_query, conn)
                
                # 获取影子模型预测
                shadow_query = """
                    SELECT prediction, timestamp FROM prediction_logs 
                    WHERE model_name = ? AND model_type = 'challenger'
                    AND timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp DESC LIMIT 100
                """
                shadow_predictions = pd.read_sql_query(shadow_query, conn, params=[challenger_name])
                
                if len(champion_predictions) > 0 and len(shadow_predictions) > 0:
                    # 计算预测一致性（简化版本）
                    # 实际实现需要根据具体的预测格式进行调整
                    agreement_count = 0
                    total_comparisons = min(len(champion_predictions), len(shadow_predictions))
                    
                    for i in range(total_comparisons):
                        # 这里需要根据实际预测格式实现比较逻辑
                        # 暂时使用简化的字符串比较
                        if champion_predictions.iloc[i]['prediction'] == shadow_predictions.iloc[i]['prediction']:
                            agreement_count += 1
                    
                    comparison_metrics['prediction_agreement_rate'] = agreement_count / total_comparisons
                    
        except Exception as e:
            logger.error(f"Failed to compare shadow predictions: {e}")
        
        return comparison_metrics
    
    def analyze_shadow_test_results(self, shadow_test: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析影子测试结果
        
        Args:
            shadow_test: 影子测试信息
            
        Returns:
            分析结果
        """
        challenger_name = shadow_test['challenger_name']
        logger.info(f"Analyzing shadow test results for {challenger_name}")
        
        analysis_result = {
            'challenger_name': challenger_name,
            'test_duration': shadow_test.get('shadow_mode_duration', 0),
            'metrics_summary': {},
            'performance_comparison': {},
            'recommendation': 'reject',
            'confidence_score': 0.0,
            'reasons': []
        }
        
        try:
            # 收集测试期间的所有指标
            metrics_collected = shadow_test.get('metrics_collected', [])
            
            if not metrics_collected:
                analysis_result['reasons'].append("No metrics collected during shadow test")
                return analysis_result
            
            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(metrics_collected)
            analysis_result['metrics_summary'] = avg_metrics
            
            # 获取冠军模型的基准指标
            champion_metrics = self._get_champion_baseline_metrics()
            
            # 比较性能
            performance_comparison = self._compare_performance(avg_metrics, champion_metrics)
            analysis_result['performance_comparison'] = performance_comparison
            
            # 决策逻辑
            decision_result = self._make_promotion_decision(performance_comparison)
            analysis_result.update(decision_result)
            
            logger.info(f"Shadow test analysis completed: {analysis_result['recommendation']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Shadow test analysis failed: {e}")
            analysis_result['reasons'].append(f"Analysis failed: {str(e)}")
            return analysis_result
    
    def _calculate_average_metrics(self, metrics_collected: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算平均指标"""
        if not metrics_collected:
            return {}
        
        # 提取数值指标
        numeric_metrics = {}
        for metric_entry in metrics_collected:
            metrics = metric_entry.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # 计算平均值
        avg_metrics = {}
        for key, values in numeric_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _get_champion_baseline_metrics(self) -> Dict[str, float]:
        """获取冠军模型基准指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT metric_name, AVG(metric_value) as avg_value
                    FROM model_metrics 
                    WHERE model_type = 'champion' 
                    AND timestamp > datetime('now', '-7 days')
                    GROUP BY metric_name
                """
                result = conn.execute(query).fetchall()
                
                baseline_metrics = {}
                for row in result:
                    baseline_metrics[row[0]] = row[1]
                
                return baseline_metrics
                
        except Exception as e:
            logger.error(f"Failed to get champion baseline metrics: {e}")
            return {}
    
    def _compare_performance(self, challenger_metrics: Dict[str, float], 
                           champion_metrics: Dict[str, float]) -> Dict[str, Any]:
        """比较性能指标"""
        comparison = {}
        
        for metric_name in self.cc_config['evaluation_metrics']:
            if metric_name in challenger_metrics and metric_name in champion_metrics:
                challenger_value = challenger_metrics[metric_name]
                champion_value = champion_metrics[metric_name]
                
                improvement = (challenger_value - champion_value) / champion_value
                
                comparison[metric_name] = {
                    'challenger': challenger_value,
                    'champion': champion_value,
                    'improvement': improvement,
                    'significant': improvement > self.cc_config['champion_threshold']
                }
        
        return comparison
    
    def _make_promotion_decision(self, performance_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """做出提升决策"""
        if not performance_comparison:
            return {
                'recommendation': 'reject',
                'confidence_score': 0.0,
                'reasons': ['No performance comparison data available']
            }
        
        significant_improvements = 0
        total_metrics = len(performance_comparison)
        positive_improvements = 0
        reasons = []
        
        for metric_name, comparison in performance_comparison.items():
            improvement = comparison['improvement']
            
            if improvement > 0:
                positive_improvements += 1
                
            if comparison['significant']:
                significant_improvements += 1
                reasons.append(f"Significant improvement in {metric_name}: {improvement:.2%}")
            elif improvement > 0:
                reasons.append(f"Positive improvement in {metric_name}: {improvement:.2%}")
            else:
                reasons.append(f"Decline in {metric_name}: {improvement:.2%}")
        
        # 决策逻辑
        if significant_improvements >= total_metrics * 0.5:
            recommendation = 'promote'
            confidence_score = 0.9
        elif positive_improvements >= total_metrics * 0.7:
            recommendation = 'promote'
            confidence_score = 0.7
        else:
            recommendation = 'reject'
            confidence_score = 0.3
        
        return {
            'recommendation': recommendation,
            'confidence_score': confidence_score,
            'reasons': reasons
        }
    
    def _get_historical_metrics(self, model_name: str, model_version: str, hours: int = 24) -> List[Dict[str, Any]]:
        """获取历史指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM prediction_logs 
                    WHERE model_name = ? AND model_version = ?
                    AND timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                """.format(hours)
                
                cursor = conn.execute(query, [model_name, model_version])
                columns = [description[0] for description in cursor.description]
                
                metrics = []
                for row in cursor.fetchall():
                    metric = dict(zip(columns, row))
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return []
    
    def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """存储指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and key not in ['timestamp']:
                        conn.execute("""
                            INSERT INTO model_metrics 
                            (model_name, model_version, model_type, metric_name, metric_value, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, [
                            metrics['model_name'],
                            metrics['model_version'],
                            metrics['model_type'],
                            key,
                            value,
                            json.dumps({'endpoint': metrics.get('endpoint')})
                        ])
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def _update_prometheus_metrics(self, metrics: Dict[str, Any]) -> None:
        """更新Prometheus指标"""
        try:
            labels = [
                metrics['model_name'],
                metrics['model_version'],
                metrics['model_type']
            ]
            
            # 更新响应时间
            if metrics.get('response_time'):
                self.metrics['prediction_latency'].labels(*labels).observe(metrics['response_time'])
            
            # 更新错误率
            if 'error_rate' in metrics:
                # 这里需要根据实际情况调整
                pass
                
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'alerts': []
        }
        
        try:
            # 检查数据库连接
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1").fetchone()
                health_status['components']['database'] = 'healthy'
                
        except Exception as e:
            health_status['components']['database'] = 'unhealthy'
            health_status['alerts'].append(f"Database connection failed: {e}")
            health_status['overall_status'] = 'degraded'
        
        return health_status

#!/usr/bin/env python3
"""
训练新挑战者模型的脚本
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger
import yaml
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lifecycle import ChampionChallengerManager


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()
    logger.add(
        log_dir / "train_challenger.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train a new challenger model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--data-path", 
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--model-name", 
        type=str,
        help="Custom model name (optional)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--auto-evaluate", 
        action="store_true",
        help="Automatically evaluate the challenger after training"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Starting Challenger Model Training")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Auto evaluate: {args.auto_evaluate}")
    logger.info("=" * 60)
    
    try:
        # 检查配置文件
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # 检查数据文件
        data_path = Path(args.data_path)
        if not data_path.exists():
            logger.error(f"Training data file not found: {data_path}")
            return 1
        
        # 初始化生命周期管理器
        logger.info("Initializing Champion-Challenger Manager")
        cc_manager = ChampionChallengerManager(str(config_path))
        
        # 训练挑战者模型
        logger.info("Starting challenger training")
        training_result = cc_manager.train_challenger(
            data_path=str(data_path),
            model_name=args.model_name
        )
        
        logger.info("Challenger training completed successfully")
        logger.info(f"Model name: {training_result['name']}")
        logger.info(f"Model version: {training_result['version']}")
        logger.info(f"Run ID: {training_result['run_id']}")
        
        # 如果启用自动评估
        if args.auto_evaluate:
            logger.info("Starting automatic evaluation")
            evaluation_result = cc_manager.evaluate_challenger(training_result['name'])
            
            logger.info("Evaluation completed")
            logger.info(f"Action taken: {evaluation_result.get('action', 'unknown')}")
            
            if 'comparison' in evaluation_result:
                comparison = evaluation_result['comparison']
                logger.info("Performance comparison:")
                for metric, details in comparison.get('improvements', {}).items():
                    improvement = details.get('improvement', 0)
                    significant = details.get('significant', False)
                    logger.info(f"  {metric}: {improvement:.2%} {'(significant)' if significant else ''}")
        
        logger.info("Process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Challenger training failed: {e}")
        logger.exception("Exception details:")
        return 1
    
    finally:
        logger.info("=" * 60)
        logger.info("Challenger Model Training Finished")
        logger.info(f"End time: {datetime.now().isoformat()}")
        logger.info("=" * 60)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

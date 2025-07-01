#!/usr/bin/env python3
"""
无Docker依赖的挑战者模型训练脚本
"""

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    # 添加文件日志
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "challenger_training.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def train_with_simple_trainer(data_path: str, config: dict, model_name: str = None, auto_evaluate: bool = False):
    """使用简化训练器训练模型"""
    try:
        # 导入简化训练器
        from src.training.ludwig_trainer import LudwigTrainer
        
        # 创建训练器
        trainer = LudwigTrainer("config/ludwig_config.yaml")
        
        # 训练模型
        model_info = trainer.train_challenger(
            data_path=data_path,
            model_name=model_name or f"challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info(f"Model training completed: {model_info}")
        
        if auto_evaluate:
            # 简化的评估逻辑
            logger.info("Auto-evaluation enabled, but skipping complex evaluation due to Docker unavailability")
            logger.info("Model has been trained and logged to MLflow")
        
        return True
        
    except Exception as e:
        logger.error(f"Training with Ludwig failed: {e}")
        logger.info("Falling back to simple sklearn training...")
        
        # 回退到简单训练
        return train_with_sklearn(data_path, model_name, auto_evaluate)

def train_with_sklearn(data_path: str, model_name: str = None, auto_evaluate: bool = False):
    """使用sklearn进行简单训练"""
    try:
        import subprocess
        import os
        
        # 设置环境变量
        env = os.environ.copy()
        env['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'
        
        # 构建命令
        cmd = [
            sys.executable, 
            "scripts/train_simple_challenger.py",
            "--data-path", data_path
        ]
        
        if model_name:
            cmd.extend(["--model-name", model_name])
        
        if auto_evaluate:
            cmd.append("--auto-evaluate")
        
        # 运行简化训练脚本
        logger.info("Running simplified sklearn training...")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Sklearn training completed successfully")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Sklearn training failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Sklearn training failed: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train challenger model without Docker dependencies")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--data-path", required=True, help="Training data path")
    parser.add_argument("--model-name", help="Model name")
    parser.add_argument("--auto-evaluate", action="store_true", help="Auto evaluate against champion")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Starting Challenger Model Training (No Docker)")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Auto evaluate: {args.auto_evaluate}")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
        
        # 检查数据文件
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            return 1
        
        # 尝试训练
        success = train_with_simple_trainer(
            data_path=args.data_path,
            config=config,
            model_name=args.model_name,
            auto_evaluate=args.auto_evaluate
        )
        
        if success:
            logger.info("Challenger training completed successfully")
            logger.info("=" * 60)
            logger.info("Training Summary:")
            logger.info("- Model trained and logged to MLflow")
            logger.info("- View results at: http://localhost:5001")
            logger.info("- Docker deployment skipped (Docker unavailable)")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("Challenger training failed")
            return 1
            
    except Exception as e:
        logger.error(f"Challenger training failed: {e}")
        import traceback
        logger.error(f"Exception details:\n{traceback.format_exc()}")
        return 1
    
    finally:
        logger.info("=" * 60)
        logger.info("Challenger Model Training Finished")
        logger.info(f"End time: {datetime.now().isoformat()}")
        logger.info("=" * 60)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

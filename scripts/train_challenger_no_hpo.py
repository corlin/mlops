#!/usr/bin/env python3
"""
无超参数优化的挑战者训练脚本
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train challenger without HPO")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--data-path", required=True, help="Training data path")
    parser.add_argument("--model-name", help="Model name")
    
    args = parser.parse_args()
    
    logger.info("Starting challenger training without HPO")
    
    try:
        from src.lifecycle import ChampionChallengerManager
        
        # 创建管理器
        cc_manager = ChampionChallengerManager(args.config)
        
        # 禁用超参数优化
        cc_manager.ludwig_trainer.use_hpo = False
        
        # 训练挑战者（不使用HPO）
        result = cc_manager.train_challenger(
            data_path=args.data_path,
            model_name=args.model_name,
            use_hpo=False  # 明确禁用HPO
        )
        
        logger.info(f"Training completed: {result}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

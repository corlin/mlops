#!/usr/bin/env python3
"""
运行冠军挑战者生命周期循环的主脚本
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
    
    logger.remove()  # 移除默认处理器
    
    # 添加文件处理器
    logger.add(
        log_dir / "lifecycle_cycle.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run Champion-Challenger Lifecycle Cycle")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--data-path", 
        type=str,
        help="Path to new training data (optional)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Perform a dry run without making changes"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Starting Champion-Challenger Lifecycle Cycle")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    
    try:
        # 检查配置文件
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # 初始化生命周期管理器
        logger.info("Initializing Champion-Challenger Manager")
        cc_manager = ChampionChallengerManager(str(config_path))
        
        # 获取当前状态
        logger.info("Getting current system status")
        current_status = cc_manager.get_status()
        logger.info(f"Current champion: {current_status.get('champion_model', {}).get('name', 'None')}")
        logger.info(f"Active challengers: {len(current_status.get('challenger_models', []))}")
        logger.info(f"Active shadow tests: {len(current_status.get('active_shadow_tests', []))}")
        
        if args.dry_run:
            logger.info("Dry run mode - no changes will be made")
            logger.info("Current system status:")
            logger.info(f"Champion model: {current_status.get('champion_model')}")
            logger.info(f"Challenger models: {current_status.get('challenger_models')}")
            logger.info(f"Shadow tests: {current_status.get('active_shadow_tests')}")
            return 0
        
        # 运行生命周期循环
        logger.info("Running lifecycle cycle")
        cycle_results = cc_manager.run_lifecycle_cycle(data_path=args.data_path)
        
        # 记录结果
        logger.info("Lifecycle cycle completed")
        logger.info(f"Actions taken: {len(cycle_results.get('actions_taken', []))}")
        logger.info(f"Errors encountered: {len(cycle_results.get('errors', []))}")
        
        # 详细记录结果
        for action in cycle_results.get('actions_taken', []):
            logger.info(f"Action: {action.get('action')} - Status: Success")
        
        for error in cycle_results.get('errors', []):
            logger.error(f"Error in {error.get('action')}: {error.get('error')}")
        
        # 获取更新后的状态
        final_status = cc_manager.get_status()
        logger.info("Final system status:")
        logger.info(f"Champion: {final_status.get('champion_model', {}).get('name', 'None')}")
        logger.info(f"Challengers: {len(final_status.get('challenger_models', []))}")
        logger.info(f"Shadow tests: {len(final_status.get('active_shadow_tests', []))}")
        
        # 返回状态码
        if cycle_results.get('errors'):
            logger.warning("Lifecycle cycle completed with errors")
            return 2
        else:
            logger.info("Lifecycle cycle completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Lifecycle cycle failed with exception: {e}")
        logger.exception("Exception details:")
        return 1
    
    finally:
        logger.info("=" * 60)
        logger.info("Champion-Challenger Lifecycle Cycle Finished")
        logger.info(f"End time: {datetime.now().isoformat()}")
        logger.info("=" * 60)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

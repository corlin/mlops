#!/usr/bin/env python3
"""
Ray兼容性检查和修复脚本
"""

import sys
import subprocess
from pathlib import Path

def check_ray_installation():
    """检查Ray安装状态"""
    print("🔍 检查Ray安装状态...")
    
    try:
        import ray
        version = ray.__version__
        print(f"   Ray版本: {version}")
        return True, version
    except ImportError:
        print("   ❌ Ray未安装")
        return False, None

def check_ray_air_checkpoint():
    """检查ray.air.Checkpoint是否可用"""
    print("🔍 检查ray.air.Checkpoint...")
    
    try:
        from ray.air import Checkpoint
        print("   ✅ ray.air.Checkpoint 可用")
        return True
    except ImportError as e:
        print(f"   ❌ ray.air.Checkpoint 不可用: {e}")
        return False

def check_ludwig_ray_integration():
    """检查Ludwig与Ray的集成"""
    print("🔍 检查Ludwig与Ray集成...")
    
    try:
        from ludwig.hyperopt.execution import get_build_hyperopt_executor, RayTuneExecutor
        print("   ✅ Ludwig Ray集成可用")
        return True
    except ImportError as e:
        print(f"   ❌ Ludwig Ray集成不可用: {e}")
        return False

def suggest_ray_fixes():
    """建议Ray修复方案"""
    print("\n💡 Ray兼容性修复建议:")
    print("=" * 50)
    
    print("1. 🔧 更新Ray到兼容版本:")
    print("   pip install 'ray[tune]>=2.0.0,<2.8.0'")
    print("   - 安装包含Tune的Ray版本")
    print("   - 避免最新版本的API变更")
    
    print("\n2. 🔄 重新安装Ludwig:")
    print("   pip uninstall ludwig -y")
    print("   pip install ludwig==0.8.5")
    print("   - 使用稳定版本的Ludwig")
    
    print("\n3. 🚫 禁用超参数优化:")
    print("   python scripts/train_challenger_no_hpo.py")
    print("   - 跳过Ray依赖的超参数优化")
    print("   - 使用标准训练流程")
    
    print("\n4. 🎯 使用简化训练:")
    print("   python scripts/train_simple_challenger.py")
    print("   - 完全避免Ludwig和Ray依赖")
    print("   - 使用sklearn进行训练")

def create_no_hpo_trainer():
    """创建无超参数优化的训练脚本"""
    print("\n📝 创建无HPO训练脚本...")
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path("scripts/train_challenger_no_hpo.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    print(f"   ✅ 创建了无HPO训练脚本: {script_path}")

def main():
    """主函数"""
    print("🔧 Ray兼容性检查工具")
    print("=" * 50)
    
    # 检查Ray
    ray_installed, ray_version = check_ray_installation()
    
    if ray_installed:
        checkpoint_available = check_ray_air_checkpoint()
        ludwig_integration = check_ludwig_ray_integration()
        
        # 总结
        print(f"\n📊 检查结果:")
        print(f"   Ray版本: {ray_version}")
        print(f"   Checkpoint可用: {'✅' if checkpoint_available else '❌'}")
        print(f"   Ludwig集成: {'✅' if ludwig_integration else '❌'}")
        
        if checkpoint_available and ludwig_integration:
            print("\n🎉 Ray兼容性良好！可以使用超参数优化")
            return True
        else:
            print("\n⚠️ Ray兼容性有问题")
    else:
        print("\n❌ Ray未安装")
    
    # 提供修复建议
    suggest_ray_fixes()
    
    # 创建备用脚本
    create_no_hpo_trainer()
    
    print("\n🎯 推荐解决方案:")
    print("1. 使用修复后的训练器（已自动跳过HPO）")
    print("2. 或使用: python scripts/train_challenger_no_docker.py")
    print("3. 或使用: python scripts/train_simple_challenger.py")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

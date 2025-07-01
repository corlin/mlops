#!/usr/bin/env python3
"""
环境修复脚本 - 解决PyTorch、Ludwig等依赖问题
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_pytorch():
    """检查PyTorch版本"""
    try:
        import torch
        version = torch.__version__
        print(f"PyTorch version: {version}")
        
        # 检查版本是否满足要求
        major, minor = version.split('.')[:2]
        if int(major) >= 2 and int(minor) >= 1:
            print("✅ PyTorch version is compatible")
            return True
        else:
            print("⚠️ PyTorch version may be incompatible with Ludwig")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def fix_pytorch():
    """修复PyTorch版本"""
    print("🔧 Fixing PyTorch installation...")
    
    # 卸载旧版本
    print("Uninstalling old PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", check=False)
    
    # 安装兼容版本
    print("Installing compatible PyTorch...")
    success, stdout, stderr = run_command(
        "pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        check=False
    )
    
    if success:
        print("✅ PyTorch installation fixed")
        return True
    else:
        print(f"❌ Failed to install PyTorch: {stderr}")
        return False

def check_ludwig():
    """检查Ludwig"""
    try:
        import ludwig
        version = ludwig.__version__
        print(f"Ludwig version: {version}")
        return True
    except ImportError:
        print("❌ Ludwig not installed")
        return False

def install_minimal_dependencies():
    """安装最小依赖集"""
    print("📦 Installing minimal dependencies for simple training...")
    
    minimal_deps = [
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "mlflow>=2.8.0",
        "loguru>=0.7.0"
    ]
    
    for dep in minimal_deps:
        print(f"Installing {dep}...")
        success, stdout, stderr = run_command(f"pip install '{dep}'", check=False)
        if success:
            print(f"✅ {dep} installed")
        else:
            print(f"⚠️ Failed to install {dep}: {stderr}")

def create_simple_config():
    """创建简化配置"""
    print("📝 Creating simplified configuration...")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    simple_config = {
        'mlflow': {
            'tracking_uri': 'http://localhost:5001',
            'experiment_name': 'simple_champion_challenger'
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42,
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10
                },
                'logistic_regression': {
                    'max_iter': 1000
                }
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'improvement_threshold': 0.01
        }
    }
    
    import yaml
    with open(config_dir / "simple_config.yaml", 'w') as f:
        yaml.dump(simple_config, f, indent=2)
    
    print("✅ Simple configuration created")

def main():
    """主函数"""
    print("🔧 MLOps Environment Fix Tool")
    print("=" * 50)
    
    # 检查当前环境
    print("🔍 Checking current environment...")
    
    pytorch_ok = check_pytorch()
    ludwig_ok = check_ludwig()
    
    # 决定修复策略
    if not pytorch_ok or not ludwig_ok:
        print("\n⚠️ Environment issues detected!")
        print("Recommended approach: Use simplified training without Ludwig")
        
        response = input("\nDo you want to install minimal dependencies for simple training? (y/N): ")
        if response.lower() == 'y':
            install_minimal_dependencies()
            create_simple_config()
            
            print("\n✅ Environment setup completed!")
            print("\n🚀 You can now use the simplified training script:")
            print("   python scripts/train_simple_challenger.py --data-path data/raw/baseline_full.csv --auto-evaluate")
            
        else:
            print("\nAlternative: Fix PyTorch and Ludwig")
            fix_response = input("Do you want to try fixing PyTorch? (y/N): ")
            if fix_response.lower() == 'y':
                if fix_pytorch():
                    print("✅ PyTorch fixed. You may need to restart your Python session.")
                else:
                    print("❌ Failed to fix PyTorch")
    
    else:
        print("✅ Environment looks good!")
        print("You can use either the full Ludwig training or simplified training.")
    
    print("\n📋 Summary:")
    print("- For simple training (recommended): use train_simple_challenger.py")
    print("- For full Ludwig training: fix PyTorch version first")
    print("- MLflow UI: http://localhost:5001")

if __name__ == "__main__":
    main()

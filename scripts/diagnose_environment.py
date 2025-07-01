#!/usr/bin/env python3
"""
环境诊断和修复脚本
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    else:
        print("✅ Python版本满足要求")
        return True

def check_pytorch():
    """检查PyTorch"""
    try:
        import torch
        version = torch.__version__
        print(f"🔥 PyTorch版本: {version}")
        
        major, minor = version.split('.')[:2]
        if int(major) >= 2 and int(minor) >= 1:
            print("✅ PyTorch版本兼容")
            return True
        else:
            print("⚠️ PyTorch版本可能不兼容（需要>=2.1）")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_ludwig():
    """检查Ludwig"""
    try:
        import ludwig
        version = ludwig.__version__
        print(f"🎯 Ludwig版本: {version}")
        return True
    except ImportError:
        print("❌ Ludwig未安装")
        return False

def check_docker():
    """检查Docker"""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("🐳 Docker: 可用")
        return True
    except Exception as e:
        print(f"❌ Docker: 不可用 ({e})")
        return False

def check_mlflow_server():
    """检查MLflow服务器"""
    try:
        import requests
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("📊 MLflow服务器: 运行中 (http://localhost:5001)")
            return True
        else:
            print(f"⚠️ MLflow服务器: 响应异常 ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ MLflow服务器: 不可用 ({e})")
        return False

def check_environment_variables():
    """检查环境变量"""
    print("\n🔧 环境变量检查:")
    
    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://localhost:5001',
        'PYTHONPATH': None
    }
    
    for var, expected in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {value}")
        else:
            if expected:
                print(f"   ⚠️ {var}: 未设置 (建议: {expected})")
            else:
                print(f"   ℹ️ {var}: 未设置")

def suggest_solutions():
    """建议解决方案"""
    print("\n💡 解决方案建议:")
    print("=" * 50)
    
    print("1. 🚀 快速开始（推荐）:")
    print("   python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
    print("   - 使用sklearn进行训练")
    print("   - 不依赖Ludwig或Docker")
    print("   - 支持冠军挑战者逻辑")
    
    print("\n2. 🔧 修复Ludwig环境:")
    print("   pip install ludwig==0.8.5 'pydantic>=1.10.0,<2.0.0'")
    print("   - 安装兼容版本的Ludwig")
    print("   - 解决Pydantic版本冲突")
    
    print("\n3. 🐳 修复Docker:")
    print("   - 确保Docker Desktop正在运行")
    print("   - 检查Docker权限设置")
    print("   - 重启Docker服务")
    
    print("\n4. 📊 启动MLflow服务器:")
    print("   cd docker && docker-compose up -d mlflow")
    print("   - 或使用: make start")
    
    print("\n5. 🎯 完整环境设置:")
    print("   python scripts/fix_environment.py")
    print("   - 自动安装最小依赖")
    print("   - 创建简化配置")

def main():
    """主函数"""
    print("🔍 MLOps环境诊断工具")
    print("=" * 50)
    
    # 检查各个组件
    checks = [
        ("Python版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("Ludwig", check_ludwig),
        ("Docker", check_docker),
        ("MLflow服务器", check_mlflow_server)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n🔍 检查{name}:")
        results[name] = check_func()
    
    # 环境变量检查
    check_environment_variables()
    
    # 总结
    print("\n📊 诊断总结:")
    print("=" * 50)
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"可用组件: {working_components}/{total_components}")
    
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {name}")
    
    # 根据结果给出建议
    if results.get("MLflow服务器", False):
        if working_components >= 3:
            print("\n🎉 环境基本可用！")
            print("建议使用: python scripts/train_challenger_no_docker.py")
        else:
            print("\n⚠️ 环境部分可用")
            print("建议使用简化训练脚本")
    else:
        print("\n❌ 环境需要修复")
        print("请先启动MLflow服务器")
    
    # 显示解决方案
    suggest_solutions()
    
    return working_components >= 2  # 至少需要Python和MLflow

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

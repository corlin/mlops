#!/usr/bin/env python3
"""
Ray和Ludwig环境修复脚本
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def install_compatible_versions():
    """安装兼容版本的依赖"""
    print("🔧 安装兼容版本的Ray和Ludwig...")
    
    # 卸载可能冲突的包
    packages_to_remove = [
        "ray", "ludwig", "torch", "torchvision", "torchaudio"
    ]
    
    for package in packages_to_remove:
        print(f"   卸载 {package}...")
        run_command(f"pip uninstall {package} -y", check=False)
    
    # 安装兼容版本
    compatible_packages = [
        "torch>=2.1.0,<2.2.0",
        "'ray[tune]>=2.0.0,<2.8.0'",
        "ludwig==0.8.5",
        "'pydantic>=1.10.0,<2.0.0'"
    ]
    
    for package in compatible_packages:
        print(f"   安装 {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            print(f"   ✅ {package} 安装成功")
        else:
            print(f"   ❌ {package} 安装失败: {stderr}")
    
    return True

def test_installation():
    """测试安装结果"""
    print("\n🔍 测试安装结果...")
    
    tests = [
        ("Ray", "import ray; print(f'Ray: {ray.__version__}')"),
        ("Ray Tune", "from ray import tune; print('Ray Tune: OK')"),
        ("Ray Air", "from ray.air import Checkpoint; print('Ray Air: OK')"),
        ("Ludwig", "import ludwig; print(f'Ludwig: {ludwig.__version__}')"),
        ("Ludwig HPO", "from ludwig.hyperopt.execution import RayTuneExecutor; print('Ludwig HPO: OK')")
    ]
    
    results = {}
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"   ✅ {name}: 正常")
            results[name] = True
        except Exception as e:
            print(f"   ❌ {name}: 失败 - {e}")
            results[name] = False
    
    return results

def create_environment_summary():
    """创建环境总结"""
    print("\n📋 环境修复总结:")
    print("=" * 50)
    
    # 测试安装
    results = test_installation()
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n工作组件: {working_count}/{total_count}")
    
    if working_count == total_count:
        print("🎉 所有组件都正常工作！")
        print("现在可以使用完整的Ludwig训练，包括超参数优化")
        return True
    elif working_count >= 2:
        print("⚠️ 部分组件工作正常")
        print("建议使用简化训练脚本")
        return False
    else:
        print("❌ 大部分组件有问题")
        print("建议使用sklearn训练脚本")
        return False

def show_usage_recommendations(full_environment_working):
    """显示使用建议"""
    print("\n🚀 使用建议:")
    print("=" * 50)
    
    if full_environment_working:
        print("1. 完整Ludwig训练（推荐）:")
        print("   python scripts/train_challenger.py --data-path data/raw/baseline_full.csv --auto-evaluate")
        print("   - 支持超参数优化")
        print("   - 完整的Ludwig功能")
        
        print("\n2. 无Docker训练:")
        print("   python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
        print("   - 跳过Docker部署")
        print("   - 保留Ludwig训练")
    
    else:
        print("1. 无Docker训练（推荐）:")
        print("   python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
        print("   - 自动回退到sklearn")
        print("   - 稳定可靠")
        
        print("\n2. 简化sklearn训练:")
        print("   python scripts/train_simple_challenger.py --data-path data/raw/baseline_full.csv --auto-evaluate")
        print("   - 纯sklearn实现")
        print("   - 最小依赖")
    
    print("\n3. 查看结果:")
    print("   - MLflow UI: http://localhost:5001")
    print("   - Model Registry: http://localhost:8080")

def main():
    """主函数"""
    print("🔧 Ray和Ludwig环境修复工具")
    print("=" * 50)
    
    print("此工具将尝试安装兼容版本的Ray和Ludwig")
    print("这可能需要几分钟时间...")
    
    response = input("\n是否继续安装？(y/N): ")
    if response.lower() != 'y':
        print("取消安装")
        print("\n💡 替代方案:")
        print("使用无依赖训练: python scripts/train_challenger_no_docker.py")
        return 1
    
    try:
        # 安装兼容版本
        install_compatible_versions()
        
        # 测试和总结
        full_working = create_environment_summary()
        
        # 显示使用建议
        show_usage_recommendations(full_working)
        
        return 0 if full_working else 1
        
    except Exception as e:
        print(f"❌ 环境修复失败: {e}")
        print("\n💡 建议使用简化训练脚本:")
        print("python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
        return 1

if __name__ == "__main__":
    sys.exit(main())

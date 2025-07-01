#!/usr/bin/env python3
"""
依赖冲突修复脚本 - 自动解决常见的依赖冲突问题
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_package_installed(package_name):
    """检查包是否已安装"""
    success, _, _ = run_command(f"python -c 'import {package_name}'", check=False)
    return success


def get_package_version(package_name):
    """获取包版本"""
    success, stdout, _ = run_command(
        f"python -c 'import {package_name}; print({package_name}.__version__)'", 
        check=False
    )
    if success:
        return stdout.strip()
    return None


def uninstall_conflicting_packages():
    """卸载可能冲突的包"""
    print("🧹 清理可能冲突的包...")
    
    packages_to_remove = [
        'ludwig', 'mlflow', 'pydantic', 'fastapi', 'pyyaml'
    ]
    
    for package in packages_to_remove:
        if check_package_installed(package):
            print(f"   卸载 {package}...")
            success, _, stderr = run_command(f"pip uninstall {package} -y", check=False)
            if not success:
                print(f"   ⚠️ 卸载 {package} 失败: {stderr}")


def install_core_packages():
    """安装核心包的兼容版本"""
    print("📦 安装核心包的兼容版本...")
    
    # 核心包及其兼容版本 (Ludwig 0.10.4 要求 pydantic<2.0)
    core_packages = [
        "ludwig==0.10.4",
        "'pydantic>=1.10.0,<2.0.0'",
        "'pyyaml>=5.0.0,<6.0.1,!=5.4.*'",
        "mlflow>=2.8.0,<3.0.0",
        "fastapi>=0.95.0,<0.96.0",
        "uvicorn>=0.20.0,<0.30.0"
    ]
    
    for package in core_packages:
        print(f"   安装 {package}...")
        success, stdout, stderr = run_command(f"pip install {package}", check=False)
        if not success:
            print(f"   ❌ 安装 {package} 失败:")
            print(f"      {stderr}")
            return False
        else:
            print(f"   ✅ {package} 安装成功")
    
    return True


def install_remaining_packages():
    """安装其余依赖"""
    print("📦 安装其余依赖...")
    
    remaining_packages = [
        "pandas>=2.0.0,<3.0.0",
        "numpy>=1.20.0,<2.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "pyarrow>=14.0.0,<16.0.0",
        "docker>=6.0.0,<7.0.0",
        "prometheus-client>=0.15.0,<1.0.0",
        "loguru>=0.7.0,<1.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "sqlalchemy>=2.0.0,<3.0.0"
    ]
    
    for package in remaining_packages:
        print(f"   安装 {package}...")
        success, _, stderr = run_command(f"pip install {package}", check=False)
        if not success:
            print(f"   ⚠️ 安装 {package} 失败: {stderr}")
        else:
            print(f"   ✅ {package} 安装成功")


def verify_installation():
    """验证安装结果"""
    print("🔍 验证安装结果...")
    
    critical_packages = {
        'ludwig': 'ludwig',
        'mlflow': 'mlflow', 
        'pydantic': 'pydantic',
        'fastapi': 'fastapi',
        'yaml': 'yaml'
    }
    
    all_good = True
    
    for import_name, display_name in critical_packages.items():
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            print(f"   ✅ {display_name}: {version}")
        else:
            print(f"   ❌ {display_name}: 未安装")
            all_good = False
    
    return all_good


def check_compatibility():
    """检查版本兼容性"""
    print("🔍 检查版本兼容性...")
    
    try:
        # 检查Ludwig和Pydantic兼容性
        ludwig_version = get_package_version('ludwig')
        pydantic_version = get_package_version('pydantic')
        yaml_version = get_package_version('yaml')
        
        if ludwig_version and pydantic_version and yaml_version:
            from packaging import version as pkg_version
            
            ludwig_ver = pkg_version.parse(ludwig_version)
            pydantic_ver = pkg_version.parse(pydantic_version)
            yaml_ver = pkg_version.parse(yaml_version)
            
            # 检查Ludwig 0.10.x + Pydantic 1.x兼容性
            if (ludwig_ver >= pkg_version.parse("0.10.0") and
                pydantic_ver >= pkg_version.parse("1.10.0") and
                pydantic_ver < pkg_version.parse("2.0.0") and
                yaml_ver < pkg_version.parse("6.0.1")):
                print("   ✅ 版本兼容性检查通过")
                return True
            else:
                print("   ⚠️ 版本兼容性可能有问题")
                print(f"      Ludwig: {ludwig_version}")
                print(f"      Pydantic: {pydantic_version}")
                print(f"      PyYAML: {yaml_version}")
                print("   建议的兼容版本:")
                print("      Ludwig: 0.10.4")
                print("      Pydantic: >=1.10.0,<2.0.0")
                print("      PyYAML: >=5.0.0,<6.0.1,!=5.4.*")
                return False
        else:
            print("   ❌ 无法获取版本信息")
            return False
            
    except Exception as e:
        print(f"   ❌ 兼容性检查失败: {e}")
        return False


def main():
    """主函数"""
    print("🔧 MLOps依赖冲突修复工具")
    print("=" * 50)
    
    # 检查是否在虚拟环境中
    if sys.prefix == sys.base_prefix:
        print("⚠️ 建议在虚拟环境中运行此脚本")
        response = input("是否继续? (y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return False
    
    try:
        # 步骤1: 卸载冲突包
        uninstall_conflicting_packages()
        
        # 步骤2: 安装核心包
        if not install_core_packages():
            print("❌ 核心包安装失败")
            return False
        
        # 步骤3: 安装其余依赖
        install_remaining_packages()
        
        # 步骤4: 验证安装
        if not verify_installation():
            print("❌ 安装验证失败")
            return False
        
        # 步骤5: 检查兼容性
        if not check_compatibility():
            print("⚠️ 兼容性检查未通过，但基本功能应该可用")
        
        print("\n🎉 依赖修复完成!")
        print("建议运行以下命令进行最终验证:")
        print("  python scripts/check_dependencies.py")
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        return False
    except Exception as e:
        print(f"❌ 修复过程中出现错误: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

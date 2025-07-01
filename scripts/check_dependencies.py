#!/usr/bin/env python3
"""
依赖检查脚本 - 验证所有必需的依赖是否正确安装
"""

import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def check_package_version(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    检查包是否安装以及版本
    
    Args:
        package_name: 包名
        min_version: 最小版本要求
        
    Returns:
        (是否安装成功, 版本信息或错误信息)
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"版本过低: {version} < {min_version}"
        
        return True, version
    except ImportError as e:
        return False, f"导入失败: {str(e)}"
    except Exception as e:
        return False, f"检查失败: {str(e)}"


def check_system_dependencies() -> Dict[str, Tuple[bool, str]]:
    """检查系统依赖"""
    system_deps = {
        'docker': 'docker --version',
        'docker-compose': 'docker-compose --version'
    }
    
    results = {}
    for dep, cmd in system_deps.items():
        try:
            result = subprocess.run(
                cmd.split(), 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                results[dep] = (True, result.stdout.strip())
            else:
                results[dep] = (False, result.stderr.strip())
        except subprocess.TimeoutExpired:
            results[dep] = (False, "命令超时")
        except FileNotFoundError:
            results[dep] = (False, "命令未找到")
        except Exception as e:
            results[dep] = (False, f"检查失败: {str(e)}")
    
    return results


def check_python_dependencies() -> Dict[str, Tuple[bool, str]]:
    """检查Python依赖"""
    # 核心依赖及其最小版本要求
    core_deps = {
        'ludwig': '0.10.0',
        'mlflow': '2.0.0',
        'pandas': '1.5.0',
        'numpy': '1.20.0',
        'scikit-learn': '1.0.0',
        'pydantic': '2.0.0',
        'fastapi': '0.95.0',
        'uvicorn': '0.20.0',
        'docker': '6.0.0',
        'prometheus_client': '0.15.0',
        'loguru': '0.6.0',
        'pyyaml': '6.0.0',
        'sqlalchemy': '2.0.0',
        'psycopg2': None,  # 可选依赖
        'boto3': None,     # 可选依赖
    }
    
    results = {}
    for package, min_version in core_deps.items():
        # 处理包名映射
        import_name = package
        if package == 'pyyaml':
            import_name = 'yaml'
        elif package == 'scikit-learn':
            import_name = 'sklearn'
        elif package == 'prometheus_client':
            import_name = 'prometheus_client'
        elif package == 'psycopg2':
            import_name = 'psycopg2'
        
        success, info = check_package_version(import_name, min_version)
        results[package] = (success, info)
    
    return results


def check_ludwig_compatibility():
    """检查Ludwig兼容性"""
    try:
        import ludwig
        import pydantic
        import yaml

        ludwig_version = ludwig.__version__
        pydantic_version = pydantic.__version__

        print(f"\n🔍 Ludwig兼容性检查:")
        print(f"   Ludwig版本: {ludwig_version}")
        print(f"   Pydantic版本: {pydantic_version}")

        # 检查PyYAML版本
        try:
            yaml_version = yaml.__version__
            print(f"   PyYAML版本: {yaml_version}")
        except:
            yaml_version = "unknown"

        # 检查是否是兼容的版本组合
        from packaging import version as pkg_version

        ludwig_ver = pkg_version.parse(ludwig_version)
        pydantic_ver = pkg_version.parse(pydantic_version)

        # Ludwig 0.10.x 兼容性检查
        if ludwig_ver >= pkg_version.parse("0.10.0"):
            # Ludwig 0.10.4 实际上仍然要求 pydantic<2.0
            if pydantic_ver >= pkg_version.parse("2.0.0"):
                print("   ❌ 版本不兼容：Ludwig 0.10.4 要求 pydantic<2.0")
                print("   建议: pip install 'pydantic>=1.10.0,<2.0.0'")
                return False
            elif pydantic_ver >= pkg_version.parse("1.10.0"):
                # 检查PyYAML兼容性
                if yaml_version != "unknown":
                    yaml_ver = pkg_version.parse(yaml_version)
                    if yaml_ver >= pkg_version.parse("6.0.1"):
                        print("   ⚠️ PyYAML版本过高，Ludwig 0.10.4要求<6.0.1")
                        print("   建议: pip install 'pyyaml>=5.0.0,<6.0.1,!=5.4.*'")
                        return False
                    elif yaml_ver == pkg_version.parse("5.4.0") or yaml_ver == pkg_version.parse("5.4.1"):
                        print("   ⚠️ PyYAML 5.4.x版本不兼容")
                        return False

                print("   ✅ 版本兼容")
                return True
            else:
                print("   ⚠️ Pydantic版本过低，建议使用 pydantic>=1.10.0,<2.0.0")
                return False
        else:
            # Ludwig 0.8.x 兼容性检查
            if pydantic_ver < pkg_version.parse("2.0.0"):
                print("   ✅ 版本兼容（旧版本组合）")
                return True
            else:
                print("   ❌ 版本不兼容：Ludwig < 0.10.0 不支持 Pydantic 2.0+")
                return False

    except Exception as e:
        print(f"   ❌ 兼容性检查失败: {e}")
        return False


def main():
    """主函数"""
    print("🔍 MLOps系统依赖检查")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    else:
        print("✅ Python版本满足要求")
    
    # 检查系统依赖
    print("\n🐳 系统依赖检查:")
    system_results = check_system_dependencies()
    system_ok = True
    
    for dep, (success, info) in system_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {dep}: {info}")
        if not success:
            system_ok = False
    
    # 检查Python依赖
    print("\n📦 Python依赖检查:")
    python_results = check_python_dependencies()
    python_ok = True
    
    for package, (success, info) in python_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {package}: {info}")
        if not success and package in ['ludwig', 'mlflow', 'pandas', 'numpy']:
            python_ok = False
    
    # Ludwig兼容性检查
    ludwig_ok = check_ludwig_compatibility()
    
    # 检查配置文件
    print("\n📋 配置文件检查:")
    config_files = [
        'config/config.yaml',
        'config/ludwig_config.yaml',
        'requirements.txt'
    ]
    
    config_ok = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"   ✅ {config_file}: 存在")
        else:
            print(f"   ❌ {config_file}: 不存在")
            config_ok = False
    
    # 总结
    print("\n📊 检查总结:")
    print("=" * 50)
    
    all_ok = system_ok and python_ok and ludwig_ok and config_ok
    
    if all_ok:
        print("🎉 所有依赖检查通过！系统可以正常运行。")
        print("\n🚀 下一步:")
        print("   1. 运行 ./quick_start.sh 启动系统")
        print("   2. 或者运行 make setup && make start")
    else:
        print("⚠️ 发现问题，请解决以下问题后重试:")
        
        if not system_ok:
            print("\n🐳 系统依赖问题:")
            print("   - 请安装Docker和Docker Compose")
            print("   - macOS: brew install docker docker-compose")
            print("   - Ubuntu: apt-get install docker.io docker-compose")
        
        if not python_ok:
            print("\n📦 Python依赖问题:")
            print("   - 运行: pip install -r requirements.txt")
            print("   - 如果遇到冲突，尝试: pip install --upgrade -r requirements.txt")
        
        if not ludwig_ok:
            print("\n🔧 Ludwig兼容性问题:")
            print("   - 确保使用兼容的Ludwig和Pydantic版本")
            print("   - Ludwig 0.10.4+ 支持 Pydantic 2.0+")
            print("   - Ludwig 0.8.x 需要 Pydantic 1.x")
        
        if not config_ok:
            print("\n📋 配置文件问题:")
            print("   - 确保所有配置文件存在")
            print("   - 运行项目初始化脚本")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

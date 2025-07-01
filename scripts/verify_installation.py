#!/usr/bin/env python3
"""
安装验证脚本 - 验证所有组件是否正确安装和配置
"""

import sys
import importlib
from pathlib import Path


def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")
    
    critical_modules = {
        'ludwig': 'Ludwig机器学习框架',
        'mlflow': 'MLflow实验跟踪',
        'pydantic': 'Pydantic数据验证',
        'fastapi': 'FastAPI Web框架',
        'pandas': 'Pandas数据处理',
        'numpy': 'NumPy数值计算',
        'sklearn': 'Scikit-learn机器学习',
        'yaml': 'PyYAML配置解析',
        'docker': 'Docker Python客户端',
        'prometheus_client': 'Prometheus监控客户端',
        'loguru': 'Loguru日志框架'
    }
    
    success_count = 0
    for module, description in critical_modules.items():
        try:
            imported_module = importlib.import_module(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"   ✅ {description}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {description}: 导入失败 - {e}")
    
    print(f"\n导入测试结果: {success_count}/{len(critical_modules)} 成功")
    return success_count == len(critical_modules)


def test_ludwig_compatibility():
    """测试Ludwig兼容性"""
    print("\n🔍 测试Ludwig兼容性...")
    
    try:
        import ludwig
        import pydantic
        from packaging import version as pkg_version
        
        ludwig_ver = ludwig.__version__
        pydantic_ver = pydantic.__version__
        
        print(f"   Ludwig版本: {ludwig_ver}")
        print(f"   Pydantic版本: {pydantic_ver}")
        
        # 检查版本兼容性
        if (pkg_version.parse(ludwig_ver) >= pkg_version.parse("0.10.0") and
            pkg_version.parse(pydantic_ver) >= pkg_version.parse("1.10.0") and
            pkg_version.parse(pydantic_ver) < pkg_version.parse("2.0.0")):
            print("   ✅ 版本兼容性检查通过")
            return True
        else:
            print("   ❌ 版本兼容性检查失败")
            return False
            
    except Exception as e:
        print(f"   ❌ 兼容性测试失败: {e}")
        return False


def test_ludwig_basic_functionality():
    """测试Ludwig基本功能"""
    print("\n🔍 测试Ludwig基本功能...")
    
    try:
        from ludwig.api import LudwigModel
        import tempfile
        import pandas as pd
        import numpy as np
        
        # 创建简单的测试数据
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # 简单的Ludwig配置
        config = {
            'input_features': [
                {'name': 'feature1', 'type': 'number'},
                {'name': 'feature2', 'type': 'category'}
            ],
            'output_features': [
                {'name': 'target', 'type': 'binary'}
            ],
            'trainer': {
                'epochs': 1,  # 只训练1个epoch用于测试
                'batch_size': 32
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 初始化模型
            model = LudwigModel(config=config, logging_level='ERROR')
            print("   ✅ Ludwig模型初始化成功")
            
            # 测试训练（快速测试）
            train_stats, _, _ = model.train(
                dataset=test_data,
                output_directory=temp_dir,
                skip_save_model=True,
                skip_save_progress=True,
                skip_save_log=True
            )
            print("   ✅ Ludwig训练功能正常")
            
            # 测试预测
            predictions = model.predict(test_data.head(5))
            print("   ✅ Ludwig预测功能正常")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Ludwig功能测试失败: {e}")
        return False


def test_mlflow_functionality():
    """测试MLflow基本功能"""
    print("\n🔍 测试MLflow基本功能...")

    try:
        import mlflow
        import tempfile
        import requests

        # 首先测试MLflow服务器连接（端口5001）
        mlflow_uri = "http://localhost:5001"
        try:
            response = requests.get(f"{mlflow_uri}/health", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ MLflow服务器连接成功 ({mlflow_uri})")
                server_available = True
            else:
                print(f"   ⚠️ MLflow服务器响应异常: {response.status_code}")
                server_available = False
        except Exception as e:
            print(f"   ⚠️ 无法连接到MLflow服务器: {e}")
            server_available = False

        # 测试基本MLflow功能（使用临时目录）
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.set_tracking_uri(f"file://{temp_dir}")

            # 测试实验创建
            experiment_id = mlflow.create_experiment("test_experiment")
            print("   ✅ MLflow实验创建成功")

            # 测试运行记录
            mlflow.set_experiment(experiment_id=experiment_id)
            with mlflow.start_run():
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 0.95)
                print("   ✅ MLflow参数和指标记录成功")

        # 如果服务器可用，测试与实际服务器的连接
        if server_available:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                experiments = mlflow.search_experiments()
                print(f"   ✅ MLflow服务器API测试成功 (找到 {len(experiments)} 个实验)")
            except Exception as e:
                print(f"   ⚠️ MLflow服务器API测试失败: {e}")

        return True

    except Exception as e:
        print(f"   ❌ MLflow功能测试失败: {e}")
        return False


def test_configuration_files():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    config_files = [
        'config/config.yaml',
        'config/ludwig_config.yaml',
        'requirements.txt',
        'requirements-flexible.txt'
    ]
    
    success_count = 0
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                if config_file.endswith('.yaml'):
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                print(f"   ✅ {config_file}: 存在且格式正确")
                success_count += 1
            except Exception as e:
                print(f"   ❌ {config_file}: 格式错误 - {e}")
        else:
            print(f"   ❌ {config_file}: 文件不存在")
    
    return success_count == len(config_files)


def test_project_structure():
    """测试项目结构"""
    print("\n🔍 测试项目结构...")
    
    required_dirs = [
        'src/data',
        'src/training', 
        'src/tracking',
        'src/lifecycle',
        'src/deployment',
        'src/monitoring',
        'scripts',
        'config',
        'docker'
    ]
    
    success_count = 0
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✅ {directory}: 存在")
            success_count += 1
        else:
            print(f"   ❌ {directory}: 不存在")
    
    return success_count == len(required_dirs)


def main():
    """主函数"""
    print("🚀 MLOps系统安装验证")
    print("=" * 50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("Ludwig兼容性测试", test_ludwig_compatibility),
        ("配置文件测试", test_configuration_files),
        ("项目结构测试", test_project_structure),
        ("MLflow功能测试", test_mlflow_functionality),
        ("Ludwig功能测试", test_ludwig_basic_functionality)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: 通过")
            else:
                print(f"❌ {test_name}: 失败")
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {e}")
        print()
    
    # 总结
    print("=" * 50)
    print(f"📊 验证结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！系统已准备就绪。")
        print("\n🚀 下一步:")
        print("   1. 运行 ./quick_start.sh 启动完整系统")
        print("   2. 或者运行 make start 启动服务")
        print("   3. 访问 http://localhost:5001 查看MLflow UI")
        return True
    else:
        print("⚠️ 部分测试失败，请检查并修复问题。")
        print("\n🔧 建议:")
        print("   1. 运行 python scripts/fix_dependencies.py 修复依赖")
        print("   2. 检查 DEPENDENCIES.md 获取详细说明")
        print("   3. 确保所有配置文件存在")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

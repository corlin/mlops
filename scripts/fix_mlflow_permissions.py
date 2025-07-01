#!/usr/bin/env python3
"""
MLflow权限和路径修复脚本
"""

import os
import sys
from pathlib import Path
import shutil

def check_mlflow_directories():
    """检查MLflow目录状态"""
    print("🔍 检查MLflow目录状态...")
    
    directories_to_check = [
        "/mlflow",
        "/mlflow/artifacts",
        "mlflow_artifacts",
        "mlruns"
    ]
    
    for dir_path in directories_to_check:
        path = Path(dir_path)
        if path.exists():
            try:
                # 测试写入权限
                test_file = path / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                print(f"   ✅ {dir_path}: 存在且可写")
            except Exception as e:
                print(f"   ❌ {dir_path}: 存在但不可写 - {e}")
        else:
            print(f"   ℹ️ {dir_path}: 不存在")

def setup_local_mlflow_directories():
    """设置本地MLflow目录"""
    print("\n🔧 设置本地MLflow目录...")
    
    project_root = Path.cwd()
    
    # 创建本地MLflow目录
    directories = [
        project_root / "mlflow_artifacts",
        project_root / "mlruns",
        project_root / "logs"
    ]
    
    for directory in directories:
        try:
            directory.mkdir(exist_ok=True)
            print(f"   ✅ 创建目录: {directory}")
            
            # 测试写入权限
            test_file = directory / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            print(f"   ✅ 写入权限正常: {directory}")
            
        except Exception as e:
            print(f"   ❌ 创建目录失败: {directory} - {e}")

def fix_environment_variables():
    """修复环境变量"""
    print("\n🔧 设置环境变量...")
    
    project_root = Path.cwd()
    
    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://localhost:5001',
        'MLFLOW_DEFAULT_ARTIFACT_ROOT': str(project_root / "mlflow_artifacts"),
        'MLFLOW_BACKEND_STORE_URI': str(project_root / "mlruns")
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   ✅ {var}={value}")

def create_mlflow_config():
    """创建MLflow配置文件"""
    print("\n📝 创建MLflow配置...")
    
    config_content = """
# MLflow环境配置
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlflow_artifacts
export MLFLOW_BACKEND_STORE_URI=./mlruns

# 确保目录存在
mkdir -p mlflow_artifacts
mkdir -p mlruns
mkdir -p logs
"""
    
    config_file = Path("mlflow_env.sh")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"   ✅ 创建配置文件: {config_file}")
    print("   💡 使用方法: source mlflow_env.sh")

def test_mlflow_setup():
    """测试MLflow设置"""
    print("\n🧪 测试MLflow设置...")
    
    try:
        import mlflow
        
        # 设置tracking URI
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"   ✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # 测试创建实验
        experiment_name = "test_permissions"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"   ✅ 创建测试实验成功: {experiment_id}")
        except Exception:
            # 实验可能已存在
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                print(f"   ✅ 测试实验已存在: {experiment.experiment_id}")
        
        # 测试运行记录
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            print("   ✅ 测试运行记录成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ MLflow测试失败: {e}")
        return False

def show_recommendations():
    """显示建议"""
    print("\n💡 建议和解决方案:")
    print("=" * 50)
    
    print("1. 使用本地MLflow存储:")
    print("   - 避免写入系统目录")
    print("   - 使用项目目录下的mlflow_artifacts")
    
    print("\n2. 设置正确的环境变量:")
    print("   export MLFLOW_TRACKING_URI=http://localhost:5001")
    print("   export MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlflow_artifacts")
    
    print("\n3. 使用无Docker训练脚本:")
    print("   python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
    print("   - 自动处理MLflow配置")
    print("   - 避免权限问题")
    
    print("\n4. 如果问题持续存在:")
    print("   - 检查Docker容器的MLflow配置")
    print("   - 确保MLflow服务器正常运行")
    print("   - 使用简化训练脚本")

def main():
    """主函数"""
    print("🔧 MLflow权限和路径修复工具")
    print("=" * 50)
    
    # 检查当前状态
    check_mlflow_directories()
    
    # 设置本地目录
    setup_local_mlflow_directories()
    
    # 修复环境变量
    fix_environment_variables()
    
    # 创建配置文件
    create_mlflow_config()
    
    # 测试设置
    mlflow_working = test_mlflow_setup()
    
    # 显示建议
    show_recommendations()
    
    print("\n🎯 总结:")
    if mlflow_working:
        print("✅ MLflow权限问题已修复")
        print("现在可以正常使用MLflow功能")
    else:
        print("⚠️ MLflow仍有问题")
        print("建议使用无Docker训练脚本")
    
    return mlflow_working

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

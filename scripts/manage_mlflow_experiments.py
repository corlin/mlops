#!/usr/bin/env python3
"""
MLflow实验管理工具
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_mlflow():
    """设置MLflow"""
    try:
        import mlflow
        
        # 设置tracking URI
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"MLflow tracking URI: {tracking_uri}")
        return mlflow
        
    except ImportError:
        print("❌ MLflow未安装")
        return None
    except Exception as e:
        print(f"❌ MLflow设置失败: {e}")
        return None

def list_experiments(mlflow):
    """列出所有实验"""
    print("\n📋 MLflow实验列表:")
    print("=" * 60)
    
    try:
        experiments = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        
        if not experiments:
            print("   没有找到实验")
            return
        
        for exp in experiments:
            status_icon = "🗑️" if exp.lifecycle_stage == 'deleted' else "✅"
            print(f"   {status_icon} ID: {exp.experiment_id}")
            print(f"      名称: {exp.name}")
            print(f"      状态: {exp.lifecycle_stage}")
            print(f"      创建时间: {exp.creation_time}")
            print()
            
    except Exception as e:
        print(f"   ❌ 获取实验列表失败: {e}")

def restore_experiment(mlflow, experiment_name):
    """恢复删除的实验"""
    print(f"\n🔄 恢复实验: {experiment_name}")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            print(f"   ❌ 实验 '{experiment_name}' 不存在")
            return False
        
        if experiment.lifecycle_stage != 'deleted':
            print(f"   ℹ️ 实验 '{experiment_name}' 未被删除")
            return True
        
        # 恢复实验
        mlflow.restore_experiment(experiment.experiment_id)
        print(f"   ✅ 实验 '{experiment_name}' 已恢复")
        return True
        
    except Exception as e:
        print(f"   ❌ 恢复实验失败: {e}")
        return False

def delete_experiment_permanently(mlflow, experiment_name):
    """永久删除实验"""
    print(f"\n🗑️ 永久删除实验: {experiment_name}")
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            print(f"   ❌ 实验 '{experiment_name}' 不存在")
            return False
        
        if experiment.lifecycle_stage != 'deleted':
            print(f"   ⚠️ 实验 '{experiment_name}' 未被标记为删除")
            print("   需要先删除实验才能永久删除")
            return False
        
        # 永久删除实验
        mlflow.delete_experiment(experiment.experiment_id)
        print(f"   ✅ 实验 '{experiment_name}' 已永久删除")
        return True
        
    except Exception as e:
        print(f"   ❌ 永久删除实验失败: {e}")
        return False

def create_experiment(mlflow, experiment_name):
    """创建新实验"""
    print(f"\n➕ 创建实验: {experiment_name}")
    
    try:
        # 检查实验是否已存在
        existing_experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if existing_experiment:
            if existing_experiment.lifecycle_stage == 'deleted':
                print(f"   ⚠️ 实验 '{experiment_name}' 已存在但被删除")
                print("   请先永久删除或恢复该实验")
                return False
            else:
                print(f"   ℹ️ 实验 '{experiment_name}' 已存在")
                return True
        
        # 创建新实验
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"   ✅ 实验 '{experiment_name}' 已创建 (ID: {experiment_id})")
        return True
        
    except Exception as e:
        print(f"   ❌ 创建实验失败: {e}")
        return False

def fix_common_experiments(mlflow):
    """修复常见的实验问题"""
    print("\n🔧 修复常见实验问题:")
    print("=" * 60)
    
    common_experiments = [
        "challenger_training",
        "ludwig_training", 
        "simple_champion_challenger"
    ]
    
    for exp_name in common_experiments:
        print(f"\n检查实验: {exp_name}")
        
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            
            if not experiment:
                print(f"   ➕ 创建缺失的实验: {exp_name}")
                create_experiment(mlflow, exp_name)
                
            elif experiment.lifecycle_stage == 'deleted':
                print(f"   🔄 发现删除的实验: {exp_name}")
                
                # 询问用户是否恢复
                response = input(f"   是否恢复实验 '{exp_name}'? (y/N): ")
                if response.lower() == 'y':
                    restore_experiment(mlflow, exp_name)
                else:
                    # 创建新的实验名称
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = f"{exp_name}_{timestamp}"
                    print(f"   ➕ 创建新实验: {new_name}")
                    create_experiment(mlflow, new_name)
            else:
                print(f"   ✅ 实验正常: {exp_name}")
                
        except Exception as e:
            print(f"   ❌ 检查实验失败: {e}")

def show_usage():
    """显示使用说明"""
    print("\n💡 使用说明:")
    print("=" * 60)
    
    print("1. 查看所有实验:")
    print("   python scripts/manage_mlflow_experiments.py list")
    
    print("\n2. 恢复删除的实验:")
    print("   python scripts/manage_mlflow_experiments.py restore <experiment_name>")
    
    print("\n3. 创建新实验:")
    print("   python scripts/manage_mlflow_experiments.py create <experiment_name>")
    
    print("\n4. 修复常见问题:")
    print("   python scripts/manage_mlflow_experiments.py fix")
    
    print("\n5. 推荐的训练命令:")
    print("   python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")

def main():
    """主函数"""
    print("🔧 MLflow实验管理工具")
    print("=" * 60)
    
    # 设置MLflow
    mlflow = setup_mlflow()
    if not mlflow:
        return 1
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        command = "list"
    else:
        command = sys.argv[1]
    
    if command == "list":
        list_experiments(mlflow)
        
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ 请提供实验名称")
            print("用法: python scripts/manage_mlflow_experiments.py restore <experiment_name>")
            return 1
        experiment_name = sys.argv[2]
        restore_experiment(mlflow, experiment_name)
        
    elif command == "create":
        if len(sys.argv) < 3:
            print("❌ 请提供实验名称")
            print("用法: python scripts/manage_mlflow_experiments.py create <experiment_name>")
            return 1
        experiment_name = sys.argv[2]
        create_experiment(mlflow, experiment_name)
        
    elif command == "fix":
        fix_common_experiments(mlflow)
        
    else:
        print(f"❌ 未知命令: {command}")
        show_usage()
        return 1
    
    show_usage()
    return 0

if __name__ == "__main__":
    sys.exit(main())

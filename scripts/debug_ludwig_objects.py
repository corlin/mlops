#!/usr/bin/env python3
"""
Ludwig对象结构调试工具
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def inspect_object(obj, name="object", max_depth=2, current_depth=0):
    """递归检查对象结构"""
    indent = "  " * current_depth
    print(f"{indent}{name}: {type(obj).__name__}")
    
    if current_depth >= max_depth:
        return
    
    # 检查对象属性
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if not attr_name.startswith('_'):  # 跳过私有属性
                if isinstance(attr_value, (dict, list)):
                    print(f"{indent}  {attr_name}: {type(attr_value).__name__} (len={len(attr_value)})")
                    if isinstance(attr_value, dict) and len(attr_value) <= 5:
                        for k, v in attr_value.items():
                            print(f"{indent}    {k}: {type(v).__name__}")
                elif isinstance(attr_value, (int, float, str, bool)):
                    print(f"{indent}  {attr_name}: {attr_value}")
                else:
                    inspect_object(attr_value, f"{attr_name}", max_depth, current_depth + 1)

def create_mock_training_stats():
    """创建模拟的TrainingStats对象"""
    print("🔍 创建模拟TrainingStats对象...")
    
    try:
        from ludwig.utils.trainer_utils import TrainingStats
        
        # 创建模拟数据
        mock_data = {
            'training': {
                'target': {
                    'accuracy': [0.6, 0.7, 0.8, 0.85],
                    'loss': [1.2, 0.9, 0.7, 0.6]
                }
            },
            'validation': {
                'target': {
                    'accuracy': [0.55, 0.65, 0.75, 0.8],
                    'loss': [1.3, 1.0, 0.8, 0.7]
                }
            }
        }
        
        # 创建TrainingStats对象
        stats = TrainingStats()
        stats.training = mock_data['training']
        stats.validation = mock_data['validation']
        
        print("✅ 模拟TrainingStats对象创建成功")
        return stats
        
    except ImportError:
        print("❌ 无法导入TrainingStats，Ludwig未安装")
        return None
    except Exception as e:
        print(f"❌ 创建TrainingStats失败: {e}")
        return None

def test_training_stats_access():
    """测试TrainingStats访问方法"""
    print("\n🔍 测试TrainingStats访问方法...")
    
    stats = create_mock_training_stats()
    if not stats:
        return
    
    print("\n1. 检查对象结构:")
    inspect_object(stats, "TrainingStats")
    
    print("\n2. 测试不同访问方法:")
    
    # 方法1: 直接访问属性
    try:
        if hasattr(stats, 'training'):
            print("   ✅ 方法1: hasattr(stats, 'training') - 成功")
            training_data = stats.training
            print(f"   训练数据类型: {type(training_data)}")
        else:
            print("   ❌ 方法1: 没有training属性")
    except Exception as e:
        print(f"   ❌ 方法1失败: {e}")
    
    # 方法2: 尝试items()方法
    try:
        for key, value in stats.items():
            print(f"   ✅ 方法2: stats.items() - {key}: {type(value)}")
    except AttributeError:
        print("   ❌ 方法2: stats.items() - AttributeError (预期的)")
    except Exception as e:
        print(f"   ❌ 方法2失败: {e}")
    
    # 方法3: __dict__访问
    try:
        if hasattr(stats, '__dict__'):
            print("   ✅ 方法3: __dict__访问")
            for key, value in stats.__dict__.items():
                print(f"     {key}: {type(value)}")
        else:
            print("   ❌ 方法3: 没有__dict__")
    except Exception as e:
        print(f"   ❌ 方法3失败: {e}")

def generate_safe_access_code():
    """生成安全的访问代码"""
    print("\n💡 推荐的安全访问代码:")
    print("=" * 50)
    
    code = '''
def safe_extract_metrics(train_stats, prefix="train"):
    """安全提取训练指标"""
    metrics = {}
    
    try:
        # 方法1: 检查training属性
        if hasattr(train_stats, 'training'):
            training_data = train_stats.training
            if isinstance(training_data, dict):
                for feature_name, feature_metrics in training_data.items():
                    if isinstance(feature_metrics, dict):
                        for metric_name, metric_values in feature_metrics.items():
                            if isinstance(metric_values, list) and metric_values:
                                # 取最后一个值
                                final_value = metric_values[-1]
                                if isinstance(final_value, (int, float)):
                                    key = f"{prefix}_{feature_name}_{metric_name}"
                                    metrics[key] = final_value
        
        # 方法2: 检查validation属性
        if hasattr(train_stats, 'validation'):
            validation_data = train_stats.validation
            if isinstance(validation_data, dict):
                for feature_name, feature_metrics in validation_data.items():
                    if isinstance(feature_metrics, dict):
                        for metric_name, metric_values in feature_metrics.items():
                            if isinstance(metric_values, list) and metric_values:
                                final_value = metric_values[-1]
                                if isinstance(final_value, (int, float)):
                                    key = f"val_{feature_name}_{metric_name}"
                                    metrics[key] = final_value
        
        # 方法3: 后备方案 - __dict__
        if not metrics and hasattr(train_stats, '__dict__'):
            for key, value in train_stats.__dict__.items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}_{key}"] = value
                    
    except Exception as e:
        print(f"Error extracting metrics: {e}")
    
    return metrics
'''
    
    print(code)

def main():
    """主函数"""
    print("🔧 Ludwig对象结构调试工具")
    print("=" * 50)
    
    # 测试TrainingStats访问
    test_training_stats_access()
    
    # 生成安全访问代码
    generate_safe_access_code()
    
    print("\n🎯 总结:")
    print("- TrainingStats对象不支持.items()方法")
    print("- 应该使用.training和.validation属性")
    print("- 指标值通常是列表，需要取最后一个值")
    print("- 建议使用try-except进行错误处理")

if __name__ == "__main__":
    main()

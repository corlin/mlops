#!/usr/bin/env python3
"""
Ludwig evaluate()方法测试工具
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_evaluate_return_values():
    """测试Ludwig evaluate()方法的返回值"""
    print("🔍 测试Ludwig evaluate()方法返回值...")
    
    try:
        from ludwig.api import LudwigModel
        print("✅ Ludwig导入成功")
        
        # 检查evaluate方法的签名
        import inspect
        sig = inspect.signature(LudwigModel.evaluate)
        print(f"evaluate方法签名: {sig}")
        
        # 检查文档字符串
        if LudwigModel.evaluate.__doc__:
            print("evaluate方法文档:")
            print(LudwigModel.evaluate.__doc__[:500] + "...")
        
    except ImportError:
        print("❌ Ludwig未安装")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False
    
    return True

def create_safe_evaluate_wrapper():
    """创建安全的evaluate包装器"""
    print("\n💡 安全的evaluate包装器代码:")
    print("=" * 50)
    
    code = '''
def safe_evaluate_model(model, dataset_path, collect_predictions=True, collect_overall_stats=True):
    """安全的模型评估包装器"""
    try:
        # 调用evaluate方法
        eval_results = model.evaluate(
            dataset=dataset_path,
            collect_predictions=collect_predictions,
            collect_overall_stats=collect_overall_stats
        )
        
        # 处理不同的返回值格式
        test_results = None
        predictions = None
        additional_data = None
        
        if isinstance(eval_results, tuple):
            num_results = len(eval_results)
            print(f"Evaluate returned {num_results} values")
            
            if num_results >= 1:
                test_results = eval_results[0]
            if num_results >= 2:
                predictions = eval_results[1]
            if num_results >= 3:
                additional_data = eval_results[2]
                
        elif isinstance(eval_results, dict):
            # 如果返回字典，直接使用
            test_results = eval_results
            
        else:
            # 其他情况
            test_results = eval_results
        
        return {
            'test_results': test_results,
            'predictions': predictions,
            'additional_data': additional_data,
            'success': True
        }
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {
            'test_results': None,
            'predictions': None,
            'additional_data': None,
            'success': False,
            'error': str(e)
        }
'''
    
    print(code)

def analyze_ludwig_versions():
    """分析Ludwig版本差异"""
    print("\n📊 Ludwig版本差异分析:")
    print("=" * 50)
    
    print("不同Ludwig版本的evaluate()返回值:")
    print("- Ludwig 0.6.x: (test_results, predictions)")
    print("- Ludwig 0.7.x: (test_results, predictions, output_directory)")
    print("- Ludwig 0.8.x: 可能返回更多值或不同结构")
    print("- Ludwig 0.9.x+: 返回值结构可能进一步变化")
    
    print("\n建议的处理策略:")
    print("1. 使用tuple解包时检查长度")
    print("2. 使用try-except处理解包错误")
    print("3. 提供后备方案")
    print("4. 记录实际返回值结构以便调试")

def generate_debug_code():
    """生成调试代码"""
    print("\n🔧 调试代码:")
    print("=" * 50)
    
    debug_code = '''
# 在Ludwig训练器中添加调试代码
def debug_evaluate_return(model, dataset_path):
    """调试evaluate返回值"""
    try:
        eval_results = model.evaluate(
            dataset=dataset_path,
            collect_predictions=True,
            collect_overall_stats=True
        )
        
        print(f"Evaluate返回类型: {type(eval_results)}")
        
        if isinstance(eval_results, tuple):
            print(f"Tuple长度: {len(eval_results)}")
            for i, item in enumerate(eval_results):
                print(f"  [{i}]: {type(item)} - {str(item)[:100]}...")
        elif isinstance(eval_results, dict):
            print(f"Dict键: {list(eval_results.keys())}")
        else:
            print(f"其他类型: {eval_results}")
            
        return eval_results
        
    except Exception as e:
        print(f"调试失败: {e}")
        return None
'''
    
    print(debug_code)

def main():
    """主函数"""
    print("🔧 Ludwig evaluate()方法测试工具")
    print("=" * 50)
    
    # 测试Ludwig导入和方法签名
    ludwig_available = test_evaluate_return_values()
    
    # 分析版本差异
    analyze_ludwig_versions()
    
    # 创建安全包装器
    create_safe_evaluate_wrapper()
    
    # 生成调试代码
    generate_debug_code()
    
    print("\n🎯 总结:")
    print("- evaluate()方法返回值在不同Ludwig版本中可能不同")
    print("- 需要使用灵活的解包策略")
    print("- 建议使用try-except处理解包错误")
    print("- 可以添加调试代码来检查实际返回值结构")
    
    if not ludwig_available:
        print("\n💡 当前Ludwig不可用，建议使用:")
        print("python scripts/train_challenger_no_docker.py --data-path data/raw/baseline_full.csv --auto-evaluate")
    
    return ludwig_available

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

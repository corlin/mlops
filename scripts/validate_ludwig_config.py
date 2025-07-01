#!/usr/bin/env python3
"""
Ludwig配置验证脚本
"""

import sys
import yaml
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def validate_basic_structure(config):
    """验证基本结构"""
    print("🔍 验证基本结构...")
    
    required_fields = ['input_features', 'output_features']
    for field in required_fields:
        if field not in config:
            print(f"   ❌ 缺少必需字段: {field}")
            return False
        else:
            print(f"   ✅ 找到字段: {field}")
    
    return True

def validate_output_features(config):
    """验证输出特征配置"""
    print("\n🔍 验证输出特征...")
    
    output_features = config.get('output_features', [])
    if not output_features:
        print("   ❌ 没有输出特征")
        return False
    
    for i, feature in enumerate(output_features):
        print(f"   检查输出特征 {i+1}: {feature.get('name', 'unnamed')}")
        
        # 检查类型
        feature_type = feature.get('type')
        if not feature_type:
            print(f"     ❌ 缺少type字段")
            return False
        
        print(f"     类型: {feature_type}")
        
        # 检查二元分类特征的特殊配置
        if feature_type == 'binary':
            preprocessing = feature.get('preprocessing', {})
            fallback_true_label = preprocessing.get('fallback_true_label')
            
            if fallback_true_label is not None:
                if isinstance(fallback_true_label, (int, float)):
                    print(f"     ❌ fallback_true_label应该是字符串，不是数字: {fallback_true_label}")
                    print(f"     💡 建议修改为: \"{fallback_true_label}\"")
                    return False
                else:
                    print(f"     ✅ fallback_true_label: {fallback_true_label}")
    
    return True

def validate_input_features(config):
    """验证输入特征配置"""
    print("\n🔍 验证输入特征...")
    
    input_features = config.get('input_features', [])
    if not input_features:
        print("   ❌ 没有输入特征")
        return False
    
    print(f"   找到 {len(input_features)} 个输入特征")
    
    for i, feature in enumerate(input_features):
        name = feature.get('name', f'feature_{i+1}')
        feature_type = feature.get('type', 'unknown')
        print(f"     {i+1}. {name} ({feature_type})")
    
    return True

def validate_with_ludwig(config):
    """使用Ludwig验证配置"""
    print("\n🔍 使用Ludwig验证配置...")
    
    try:
        from ludwig.schema.model_types.base import ModelConfig
        from ludwig.config_validation.validation import check_schema
        
        # 验证配置
        check_schema(config)
        print("   ✅ Ludwig配置验证通过")
        return True
        
    except ImportError:
        print("   ⚠️ Ludwig未安装，跳过Ludwig验证")
        return True
    except Exception as e:
        print(f"   ❌ Ludwig配置验证失败: {e}")
        return False

def suggest_fixes(config):
    """建议修复方案"""
    print("\n💡 配置修复建议:")
    print("=" * 50)
    
    # 检查常见问题
    output_features = config.get('output_features', [])
    
    for feature in output_features:
        if feature.get('type') == 'binary':
            preprocessing = feature.get('preprocessing', {})
            fallback_true_label = preprocessing.get('fallback_true_label')
            
            if isinstance(fallback_true_label, (int, float)):
                print(f"1. 修复 fallback_true_label:")
                print(f"   当前值: {fallback_true_label} (数字)")
                print(f"   建议值: \"{fallback_true_label}\" (字符串)")
                print(f"   或者: null (如果不需要)")
    
    print("\n2. 其他建议:")
    print("   - 确保所有字符串值都用引号包围")
    print("   - 数值参数可以不用引号")
    print("   - 布尔值使用 true/false")
    print("   - 空值使用 null")

def create_fixed_config(config, output_path):
    """创建修复后的配置"""
    print(f"\n🔧 创建修复后的配置: {output_path}")
    
    # 深拷贝配置
    import copy
    fixed_config = copy.deepcopy(config)
    
    # 修复已知问题
    output_features = fixed_config.get('output_features', [])
    
    for feature in output_features:
        if feature.get('type') == 'binary':
            preprocessing = feature.get('preprocessing', {})
            fallback_true_label = preprocessing.get('fallback_true_label')
            
            if isinstance(fallback_true_label, (int, float)):
                # 转换为字符串
                preprocessing['fallback_true_label'] = str(fallback_true_label)
                print(f"   修复 fallback_true_label: {fallback_true_label} -> \"{fallback_true_label}\"")
    
    # 保存修复后的配置
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(fixed_config, f, indent=2, default_flow_style=False)
        print(f"   ✅ 修复后的配置已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 Ludwig配置验证工具")
    print("=" * 50)
    
    config_path = "config/ludwig_config.yaml"
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        return 1
    
    print(f"✅ 配置文件加载成功: {config_path}")
    
    # 验证步骤
    validations = [
        ("基本结构", lambda: validate_basic_structure(config)),
        ("输入特征", lambda: validate_input_features(config)),
        ("输出特征", lambda: validate_output_features(config)),
        ("Ludwig验证", lambda: validate_with_ludwig(config))
    ]
    
    all_passed = True
    for name, validation_func in validations:
        try:
            if not validation_func():
                all_passed = False
        except Exception as e:
            print(f"   ❌ {name}验证异常: {e}")
            all_passed = False
    
    # 结果总结
    print("\n📊 验证结果:")
    if all_passed:
        print("✅ 所有验证通过！配置文件正确")
        return 0
    else:
        print("❌ 验证失败，配置文件有问题")
        
        # 提供修复建议
        suggest_fixes(config)
        
        # 创建修复后的配置
        fixed_path = "config/ludwig_config_fixed.yaml"
        if create_fixed_config(config, fixed_path):
            print(f"\n🎯 使用修复后的配置:")
            print(f"   cp {fixed_path} {config_path}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
端口检查和冲突解决脚本
"""

import socket
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def check_port_available(port: int, host: str = 'localhost') -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # 0表示连接成功，即端口被占用
    except Exception:
        return False


def get_process_using_port(port: int) -> str:
    """获取占用端口的进程信息"""
    try:
        # 在macOS和Linux上使用lsof
        result = subprocess.run(
            ['lsof', '-i', f':{port}', '-t'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split('\n')[0]
            # 获取进程详细信息
            ps_result = subprocess.run(
                ['ps', '-p', pid, '-o', 'comm='],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ps_result.returncode == 0:
                return f"PID {pid} ({ps_result.stdout.strip()})"
        
        # 如果lsof不可用，尝试netstat
        result = subprocess.run(
            ['netstat', '-tulpn'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if f':{port} ' in line:
                    return line.strip()
        
        return "未知进程"
    except Exception as e:
        return f"检查失败: {e}"


def suggest_alternative_port(base_port: int, service_name: str) -> int:
    """建议替代端口"""
    alternative_ports = {
        'postgres': [5433, 5434, 5435],
        'mlflow': [5001, 5002, 5003],
        'grafana': [3001, 3002, 3003],
        'prometheus': [9091, 9092, 9093],
        'model-registry': [8081, 8082, 8083]
    }
    
    if service_name in alternative_ports:
        for port in alternative_ports[service_name]:
            if check_port_available(port):
                return port
    
    # 如果预定义端口都不可用，从base_port+1开始查找
    for i in range(1, 100):
        test_port = base_port + i
        if check_port_available(test_port):
            return test_port
    
    return base_port + 100  # 最后的备选


def check_required_ports() -> Dict[str, Tuple[int, bool, str]]:
    """检查所需的端口"""
    required_ports = {
        'MLflow': 5000,
        'PostgreSQL': 5433,  # 已经改为5433
        'Prometheus': 9090,
        'Grafana': 3000,
        'Model Registry UI': 8080
    }
    
    results = {}
    for service, port in required_ports.items():
        available = check_port_available(port)
        process_info = "" if available else get_process_using_port(port)
        results[service] = (port, available, process_info)
    
    return results


def generate_alternative_compose_file(port_conflicts: Dict[str, int]) -> str:
    """生成替代的docker-compose文件"""
    compose_file = Path("docker/docker-compose.yml")
    
    if not compose_file.exists():
        return "docker-compose.yml文件不存在"
    
    try:
        with open(compose_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换冲突的端口
        for service, new_port in port_conflicts.items():
            if service == 'PostgreSQL':
                content = content.replace('"5433:5432"', f'"{new_port}:5432"')
            elif service == 'MLflow':
                content = content.replace('"5000:5000"', f'"{new_port}:5000"')
            elif service == 'Grafana':
                content = content.replace('"3000:3000"', f'"{new_port}:3000"')
            elif service == 'Prometheus':
                content = content.replace('"9090:9090"', f'"{new_port}:9090"')
            elif service == 'Model Registry UI':
                content = content.replace('"8080:8080"', f'"{new_port}:8080"')
        
        # 保存到新文件
        alt_file = Path("docker/docker-compose-alt.yml")
        with open(alt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(alt_file)
        
    except Exception as e:
        return f"生成失败: {e}"


def stop_conflicting_services():
    """尝试停止可能冲突的服务"""
    print("🛑 尝试停止可能冲突的服务...")
    
    # 常见的可能冲突的服务
    services_to_stop = [
        'postgresql',
        'postgres', 
        'grafana-server',
        'prometheus'
    ]
    
    for service in services_to_stop:
        try:
            # 尝试使用systemctl停止服务
            result = subprocess.run(
                ['sudo', 'systemctl', 'stop', service],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"   ✅ 已停止服务: {service}")
            else:
                print(f"   ⚠️ 无法停止服务: {service} (可能未安装或已停止)")
        except Exception:
            print(f"   ⚠️ 检查服务失败: {service}")


def main():
    """主函数"""
    print("🔍 MLOps端口冲突检查和解决")
    print("=" * 50)
    
    # 检查端口状态
    port_results = check_required_ports()
    
    conflicts = []
    print("📊 端口状态检查:")
    for service, (port, available, process_info) in port_results.items():
        if available:
            print(f"   ✅ {service} (端口 {port}): 可用")
        else:
            print(f"   ❌ {service} (端口 {port}): 被占用 - {process_info}")
            conflicts.append(service)
    
    if not conflicts:
        print("\n🎉 所有端口都可用！可以正常启动服务。")
        return True
    
    print(f"\n⚠️ 发现 {len(conflicts)} 个端口冲突")
    
    # 提供解决方案
    print("\n🔧 解决方案:")
    print("1. 停止冲突的服务（推荐）")
    print("2. 使用替代端口")
    print("3. 手动解决冲突")
    
    choice = input("\n请选择解决方案 (1/2/3): ").strip()
    
    if choice == '1':
        stop_conflicting_services()
        
        # 重新检查
        print("\n🔍 重新检查端口状态...")
        new_results = check_required_ports()
        remaining_conflicts = [
            service for service, (_, available, _) in new_results.items() 
            if not available
        ]
        
        if not remaining_conflicts:
            print("✅ 所有冲突已解决！")
            return True
        else:
            print(f"⚠️ 仍有 {len(remaining_conflicts)} 个冲突未解决")
            choice = '2'  # 自动转到方案2
    
    if choice == '2':
        print("\n🔄 生成替代端口配置...")
        
        port_alternatives = {}
        for service in conflicts:
            original_port = port_results[service][0]
            service_key = service.lower().replace(' ', '_').replace('_ui', '')
            new_port = suggest_alternative_port(original_port, service_key)
            port_alternatives[service] = new_port
            print(f"   {service}: {original_port} → {new_port}")
        
        # 生成新的compose文件
        alt_file = generate_alternative_compose_file(port_alternatives)
        if alt_file.startswith("docker/"):
            print(f"\n✅ 已生成替代配置文件: {alt_file}")
            print("\n🚀 使用以下命令启动服务:")
            print(f"   cd docker && docker-compose -f {Path(alt_file).name} up -d")
            
            # 更新配置文件中的数据库URL
            if 'PostgreSQL' in port_alternatives:
                new_db_port = port_alternatives['PostgreSQL']
                print(f"\n📝 请更新 config/config.yaml 中的数据库URL:")
                print(f"   database.url: postgresql://mlops_user:mlops_password@localhost:{new_db_port}/mlops")
        else:
            print(f"❌ 生成替代配置失败: {alt_file}")
            return False
    
    elif choice == '3':
        print("\n📋 手动解决冲突指南:")
        for service, (port, _, process_info) in port_results.items():
            if not port_results[service][1]:  # 如果端口不可用
                print(f"\n{service} (端口 {port}):")
                print(f"   占用进程: {process_info}")
                print(f"   解决方法:")
                print(f"   - 停止占用进程: sudo kill -9 <PID>")
                print(f"   - 或修改 docker/docker-compose.yml 中的端口映射")
    
    else:
        print("❌ 无效选择")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

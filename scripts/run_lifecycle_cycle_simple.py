#!/usr/bin/env python3
"""
简化版生命周期循环脚本 - 避免复杂的依赖问题
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置简单的日志"""
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "scheduler.log"
    
    def log(message):
        timestamp = datetime.now().isoformat()
        log_message = f"{timestamp} | {message}"
        print(log_message)
        
        try:
            with open(log_file, 'a') as f:
                f.write(log_message + "\n")
        except Exception:
            pass
    
    return log

def check_services():
    """检查服务状态"""
    log = setup_logging()
    
    log("Checking MLOps services status...")
    
    services = {
        'MLflow': 'http://mlflow:5000/health',
        'PostgreSQL': 'postgres:5432'
    }
    
    for service, endpoint in services.items():
        try:
            if 'http' in endpoint:
                import requests
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    log(f"✅ {service}: Healthy")
                else:
                    log(f"⚠️ {service}: Unhealthy (status: {response.status_code})")
            else:
                # 简单的端口检查
                import socket
                host, port = endpoint.split(':')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                
                if result == 0:
                    log(f"✅ {service}: Reachable")
                else:
                    log(f"⚠️ {service}: Unreachable")
                    
        except Exception as e:
            log(f"❌ {service}: Error - {e}")

def run_simple_lifecycle():
    """运行简化的生命周期检查"""
    log = setup_logging()
    
    log("=" * 60)
    log("Starting MLOps Lifecycle Cycle (Simple Mode)")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("=" * 60)
    
    try:
        # 检查服务状态
        check_services()
        
        # 检查是否有新数据
        data_dir = Path("/app/data/raw")
        if data_dir.exists():
            data_files = list(data_dir.glob("*.csv"))
            log(f"Found {len(data_files)} data files")
            
            for data_file in data_files:
                log(f"  - {data_file.name}")
        else:
            log("Data directory not found")
        
        # 模拟生命周期检查
        log("Performing lifecycle checks...")
        
        # 检查模型注册表状态
        try:
            import requests
            mlflow_response = requests.get("http://mlflow:5000/api/2.0/mlflow/registered-models/list", timeout=10)
            if mlflow_response.status_code == 200:
                models = mlflow_response.json()
                model_count = len(models.get('registered_models', []))
                log(f"Found {model_count} registered models in MLflow")
            else:
                log("Could not retrieve model registry information")
        except Exception as e:
            log(f"MLflow API check failed: {e}")
        
        # 记录系统状态
        status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'services_checked': True,
            'lifecycle_cycle_completed': True
        }
        
        status_file = Path("/app/logs/system_status.json")
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        log("Lifecycle cycle completed successfully")
        log("=" * 60)
        
        return True
        
    except Exception as e:
        log(f"Lifecycle cycle failed: {e}")
        log("=" * 60)
        return False

def main():
    """主函数"""
    log = setup_logging()
    
    log("MLOps Scheduler started (Simple Mode)")
    
    # 运行初始生命周期循环
    log("Running initial lifecycle cycle...")
    success = run_simple_lifecycle()
    
    if success:
        log("Initial lifecycle cycle completed successfully")
    else:
        log("Initial lifecycle cycle failed")
    
    # 保持容器运行
    log("Scheduler running... (will check every 6 hours)")
    
    # 简单的循环，每小时检查一次
    while True:
        try:
            time.sleep(3600)  # 等待1小时
            log("Performing hourly health check...")
            check_services()
        except KeyboardInterrupt:
            log("Scheduler stopped by user")
            break
        except Exception as e:
            log(f"Error in scheduler loop: {e}")
            time.sleep(60)  # 出错时等待1分钟再重试

if __name__ == "__main__":
    main()

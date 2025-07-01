#!/bin/bash

# MLflow启动脚本 - 确保绑定到0.0.0.0

echo "Starting MLflow server..."
echo "Installing dependencies..."

# 安装依赖
pip install -r /requirements.txt

echo "Dependencies installed successfully"

# 创建MLflow目录
mkdir -p /mlflow/artifacts

# 检查并清理可能占用端口的进程
echo "Checking for processes using port 5000..."
if netstat -tulpn 2>/dev/null | grep :5000; then
    echo "Port 5000 is in use, killing processes..."
    pkill -f "mlflow server" || true
    pkill -f "gunicorn" || true
    sleep 2
fi

echo "Starting MLflow server on 0.0.0.0:5000..."

# 启动MLflow服务器，使用更简单的配置
exec mlflow server \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts

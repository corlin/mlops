#!/bin/bash

# 端口冲突快速解决脚本

echo "🔧 MLOps端口冲突快速解决"
echo "=========================="

# 检查是否有端口冲突
echo "🔍 检查端口状态..."

# 检查PostgreSQL端口5432
if lsof -i :5432 >/dev/null 2>&1; then
    echo "⚠️ 端口5432被占用（通常是本地PostgreSQL）"
    echo "💡 解决方案：我们已将Docker PostgreSQL配置为使用端口5433"
else
    echo "✅ 端口5432可用"
fi

# 检查我们要使用的端口5433
if lsof -i :5433 >/dev/null 2>&1; then
    echo "❌ 端口5433被占用"
    echo "🛑 尝试停止占用进程..."
    
    # 获取占用进程的PID
    PID=$(lsof -t -i :5433)
    if [ ! -z "$PID" ]; then
        echo "   发现进程PID: $PID"
        echo "   尝试停止进程..."
        kill -TERM $PID 2>/dev/null
        sleep 2
        
        # 检查是否成功停止
        if lsof -i :5433 >/dev/null 2>&1; then
            echo "   ⚠️ 进程仍在运行，尝试强制停止..."
            kill -KILL $PID 2>/dev/null
            sleep 1
        fi
        
        # 最终检查
        if lsof -i :5433 >/dev/null 2>&1; then
            echo "   ❌ 无法停止进程，请手动处理"
            exit 1
        else
            echo "   ✅ 进程已停止"
        fi
    fi
else
    echo "✅ 端口5433可用"
fi

# 检查其他端口
PORTS=(5000 3000 9090 8080)
PORT_NAMES=("MLflow" "Grafana" "Prometheus" "Model Registry")

for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    NAME=${PORT_NAMES[$i]}
    
    if lsof -i :$PORT >/dev/null 2>&1; then
        echo "⚠️ 端口$PORT被占用 ($NAME)"
        
        # 对于某些服务，尝试优雅停止
        case $PORT in
            3000)
                echo "   尝试停止Grafana服务..."
                sudo systemctl stop grafana-server 2>/dev/null || echo "   (系统服务未找到)"
                ;;
            9090)
                echo "   尝试停止Prometheus服务..."
                sudo systemctl stop prometheus 2>/dev/null || echo "   (系统服务未找到)"
                ;;
        esac
        
        # 再次检查
        if lsof -i :$PORT >/dev/null 2>&1; then
            echo "   ⚠️ 端口$PORT仍被占用，可能需要手动处理"
        else
            echo "   ✅ 端口$PORT已释放"
        fi
    else
        echo "✅ 端口$PORT可用 ($NAME)"
    fi
done

echo ""
echo "🐳 检查Docker状态..."

# 检查Docker是否运行
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker未运行，请启动Docker"
    exit 1
else
    echo "✅ Docker正在运行"
fi

# 停止可能存在的旧容器
echo "🧹 清理旧容器..."
cd "$(dirname "$0")/../docker"

if docker-compose ps -q 2>/dev/null | grep -q .; then
    echo "   停止现有容器..."
    docker-compose down
fi

echo ""
echo "🎉 端口冲突检查完成！"
echo ""
echo "📋 摘要："
echo "   - PostgreSQL将使用端口5433（避免与本地PostgreSQL冲突）"
echo "   - MLflow将使用端口5000"
echo "   - Grafana将使用端口3000"
echo "   - Prometheus将使用端口9090"
echo "   - Model Registry将使用端口8080"
echo ""
echo "🚀 现在可以安全启动服务："
echo "   cd docker && docker-compose up -d"
echo "   或运行: make start"

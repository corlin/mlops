#!/bin/bash

# MLOps Champion-Challenger 快速启动脚本

set -e

echo "🚀 MLOps Champion-Challenger 系统快速启动"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python3"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p data/raw data/processed models/ludwig_output logs monitoring state

# 安装Python依赖
echo "📦 安装Python依赖..."
echo "   注意：正在安装Ludwig 0.10.4和兼容的依赖..."

# 尝试安装标准requirements.txt
echo "   尝试安装标准依赖..."
if pip3 install -r requirements.txt; then
    echo "   ✅ 标准依赖安装成功"
else
    echo "   ⚠️ 标准依赖安装失败，尝试灵活版本..."
    if pip3 install -r requirements-flexible.txt; then
        echo "   ✅ 灵活版本依赖安装成功"
    else
        echo "   ❌ 依赖安装失败，请手动解决冲突"
        echo "   建议："
        echo "   1. 创建虚拟环境: python3 -m venv mlops_env && source mlops_env/bin/activate"
        echo "   2. 升级pip: pip install --upgrade pip"
        echo "   3. 单独安装核心依赖: pip install ludwig==0.10.4 'pydantic>=2.0.0,<3.0.0' 'pyyaml>=5.0.0,<6.0.1,!=5.4.*'"
        exit 1
    fi
fi

# 验证关键依赖
echo "🔍 验证关键依赖安装..."
python3 -c "import ludwig; print(f'Ludwig版本: {ludwig.__version__}')" || {
    echo "❌ Ludwig安装失败"
    exit 1
}
python3 -c "import mlflow; print(f'MLflow版本: {mlflow.__version__}')" || {
    echo "❌ MLflow安装失败"
    exit 1
}
python3 -c "import pydantic; print(f'Pydantic版本: {pydantic.__version__}')" || {
    echo "❌ Pydantic安装失败"
    exit 1
}
python3 -c "import yaml; print(f'PyYAML版本: {yaml.__version__}')" || {
    echo "❌ PyYAML安装失败"
    exit 1
}

# 运行完整验证
echo "🔍 运行安装验证..."
python3 scripts/verify_installation.py || {
    echo "❌ 安装验证失败，请检查依赖"
    exit 1
}

# 初始化数据库
echo "🗄️ 初始化数据库..."
python3 scripts/init_database.py

# 生成示例数据
echo "📊 生成示例数据..."
python3 examples/generate_sample_data.py --samples 5000 --output-dir data/raw
python3 examples/generate_sample_data.py --samples 2000 --generate-drift --output-dir data/raw

# 检查端口冲突
echo "🔍 检查端口冲突..."
python3 scripts/check_ports.py || {
    echo "❌ 端口冲突检测失败，请手动解决"
    echo "💡 提示："
    echo "   - 运行 'python3 scripts/check_ports.py' 获取详细信息"
    echo "   - 或者停止占用端口5433的服务"
    echo "   - PostgreSQL通常使用端口5432，我们使用5433避免冲突"
    exit 1
}

# 启动Docker服务
echo "🐳 启动Docker服务..."
cd docker
docker-compose up -d

echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
cd ..

# 检查MLflow
if curl -f http://localhost:5001/health &> /dev/null; then
    echo "✅ MLflow服务正常"
else
    echo "⚠️ MLflow服务可能还在启动中"
fi

# 检查Grafana
if curl -f http://localhost:3000/api/health &> /dev/null; then
    echo "✅ Grafana服务正常"
else
    echo "⚠️ Grafana服务可能还在启动中"
fi

# 训练第一个挑战者模型
echo "🤖 训练第一个挑战者模型..."
python3 scripts/train_challenger.py --data-path data/raw/baseline_full.csv --auto-evaluate

# 运行一次生命周期循环
echo "🔄 运行生命周期循环..."
python3 scripts/run_lifecycle_cycle.py --data-path data/raw/drift_full.csv

echo ""
echo "🎉 系统启动完成！"
echo "==================="
echo ""
echo "📊 访问以下地址查看系统状态："
echo "   MLflow UI:        http://localhost:5001"
echo "   模型注册表UI:     http://localhost:8080"
echo "   Grafana监控:      http://localhost:3000 (admin/admin)"
echo "   Prometheus:       http://localhost:9090"
echo ""
echo "🔧 常用命令："
echo "   查看系统状态:     python3 scripts/run_lifecycle_cycle.py --dry-run"
echo "   训练新挑战者:     python3 scripts/train_challenger.py --data-path <数据路径>"
echo "   运行生命周期:     python3 scripts/run_lifecycle_cycle.py"
echo "   停止服务:         cd docker && docker-compose down"
echo ""
echo "📖 更多信息请查看 README.md"

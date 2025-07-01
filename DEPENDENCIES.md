# 依赖管理和版本兼容性

## 核心依赖版本

本项目使用以下核心依赖的特定版本组合，以确保兼容性：

### 主要框架
- **Ludwig**: 0.10.4 (最新稳定版，要求Pydantic<2.0)
- **MLflow**: 2.8.1 (实验跟踪和模型管理)
- **FastAPI**: 0.95.2 (API服务框架，兼容Pydantic 1.x)
- **Pydantic**: 1.10.13 (数据验证，与Ludwig 0.10.4兼容)

### 数据处理
- **Pandas**: 2.1.3
- **NumPy**: 1.24.3
- **Scikit-learn**: 1.3.2
- **PyArrow**: 14.0.1

### 部署和监控
- **Docker**: 6.1.3 (Python客户端)
- **Prometheus Client**: 0.19.0
- **Uvicorn**: 0.24.0

## 版本兼容性矩阵

| Ludwig版本 | Pydantic版本 | 兼容性 | 说明 |
|-----------|-------------|--------|------|
| 0.10.4    | 1.10.x      | ✅     | 推荐组合，当前稳定版本 |
| 0.8.x     | 1.10.x      | ✅     | 旧版本稳定组合 |
| 0.10.4    | 2.0+        | ❌     | 不兼容，Ludwig 0.10.4要求<2.0 |
| 0.8.x     | 2.0+        | ❌     | 不兼容 |

## 依赖冲突解决

### 常见冲突场景

1. **Ludwig + Pydantic 2.0 冲突**
   ```
   错误: ludwig 0.10.4 depends on pydantic<2.0
   ```

   **解决方案**:
   ```bash
   # 使用兼容的Pydantic 1.x版本
   pip install ludwig==0.10.4 'pydantic>=1.10.0,<2.0.0'
   ```

2. **PyYAML版本冲突**
   ```
   错误: ludwig 0.10.4 depends on PyYAML!=5.4.*, <6.0.1 and >=3.12
   ```

   **解决方案**:
   ```bash
   # 使用兼容的PyYAML版本
   pip install 'pyyaml>=5.0.0,<6.0.1,!=5.4.*'
   ```

3. **FastAPI + Pydantic版本冲突**
   ```
   错误: FastAPI requires pydantic>=2.0, but ludwig requires pydantic<2.0
   ```

   **解决方案**:
   ```bash
   # 使用兼容的FastAPI版本
   pip install ludwig==0.10.4 fastapi==0.95.2 'pydantic>=1.10.0,<2.0.0'
   ```

### 推荐安装流程

1. **清理环境**
   ```bash
   pip uninstall ludwig mlflow pydantic fastapi -y
   ```

2. **安装兼容版本**
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装**
   ```bash
   python scripts/check_dependencies.py
   ```

## 虚拟环境设置

为避免依赖冲突，强烈推荐使用虚拟环境：

### 使用venv
```bash
python -m venv mlops_env
source mlops_env/bin/activate  # Linux/Mac
# mlops_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 使用conda
```bash
conda create -n mlops python=3.9
conda activate mlops
pip install -r requirements.txt
```

### 使用poetry
```bash
poetry install
poetry shell
```

## 系统要求

### Python版本
- **最低要求**: Python 3.8
- **推荐版本**: Python 3.9 或 3.10
- **测试版本**: Python 3.9.18

### 系统依赖
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.0+

### 硬件要求
- **内存**: 最少8GB，推荐16GB+
- **存储**: 最少10GB可用空间
- **CPU**: 多核处理器推荐
- **GPU**: 可选，用于加速训练

## 开发环境设置

### 完整开发环境
```bash
# 1. 克隆项目
git clone <repository-url>
cd mlops2

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 检查依赖
python scripts/check_dependencies.py

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装开发工具
pip install pytest black flake8 mypy isort pre-commit

# 6. 设置pre-commit钩子
pre-commit install

# 7. 验证安装
make test
```

## 生产环境部署

### Docker部署（推荐）
```bash
# 使用Docker避免依赖问题
docker-compose up -d
```

### 手动部署
```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv docker.io docker-compose

# 2. 设置Python环境
python3.9 -m venv /opt/mlops
source /opt/mlops/bin/activate

# 3. 安装Python依赖
pip install -r requirements.txt

# 4. 启动服务
systemctl start docker
docker-compose up -d
```

## 故障排除

### 依赖检查工具
```bash
# 运行完整的依赖检查
python scripts/check_dependencies.py

# 检查特定包
python -c "import ludwig; print(ludwig.__version__)"
python -c "import pydantic; print(pydantic.__version__)"
```

### 常见错误和解决方案

1. **ModuleNotFoundError: No module named 'ludwig'**
   ```bash
   pip install ludwig==0.10.4
   ```

2. **ImportError: cannot import name 'BaseModel' from 'pydantic'**
   ```bash
   pip install pydantic==2.5.0
   ```

3. **Docker permission denied**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

## 更新指南

### 升级依赖
```bash
# 1. 备份当前环境
pip freeze > current_requirements.txt

# 2. 更新到新版本
pip install -r requirements.txt --upgrade

# 3. 测试兼容性
python scripts/check_dependencies.py
make test

# 4. 如果有问题，回滚
pip install -r current_requirements.txt
```

### 版本锁定
项目使用固定版本号以确保可重现性。如需升级：

1. 更新 `requirements.txt`
2. 测试兼容性
3. 更新文档
4. 提交更改

## 支持和帮助

如果遇到依赖问题：

1. 首先运行 `python scripts/check_dependencies.py`
2. 查看本文档的故障排除部分
3. 检查项目的GitHub Issues
4. 创建新的Issue并提供详细信息

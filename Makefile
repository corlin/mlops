# MLOps Champion-Challenger Makefile

.PHONY: help install setup start stop clean test lint format docker-build docker-up docker-down

# 默认目标
help:
	@echo "Available commands:"
	@echo "  check-deps       - Check system and Python dependencies"
	@echo "  check-ports      - Check for port conflicts and suggest solutions"
	@echo "  fix-ports        - Automatically fix port conflicts"
	@echo "  install          - Install Python dependencies (with fallback to flexible versions)"
	@echo "  install-flexible - Install flexible version dependencies"
	@echo "  fix-deps         - Automatically fix dependency conflicts"
	@echo "  verify           - Verify installation and functionality"
	@echo "  setup            - Setup project (install + init database + generate sample data)"
	@echo "  start            - Start all services"
	@echo "  stop             - Stop all services"
	@echo "  clean            - Clean up generated files"
	@echo "  test             - Run tests"
	@echo "  lint             - Run code linting"
	@echo "  format           - Format code"
	@echo "  docker-build     - Build Docker images"
	@echo "  docker-up        - Start Docker services"
	@echo "  docker-down      - Stop Docker services"

# 检查依赖
check-deps:
	@echo "Checking system and Python dependencies..."
	python3 scripts/check_dependencies.py

# 检查端口冲突
check-ports:
	@echo "Checking for port conflicts..."
	python3 scripts/check_ports.py

# 修复端口冲突
fix-ports:
	@echo "Fixing port conflicts..."
	./scripts/fix_port_conflicts.sh

# 安装依赖
install:
	@echo "Installing Python dependencies..."
	@echo "Note: Installing Ludwig 0.10.4 with compatible dependencies..."
	@if pip install -r requirements.txt; then \
		echo "✅ Standard dependencies installed successfully"; \
	else \
		echo "⚠️ Standard install failed, trying flexible versions..."; \
		pip install -r requirements-flexible.txt; \
	fi
	@echo "Verifying installation..."
	python3 -c "import ludwig; print(f'Ludwig: {ludwig.__version__}')"
	python3 -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"
	python3 -c "import yaml; print(f'PyYAML: {yaml.__version__}')"

# 强制安装灵活版本
install-flexible:
	@echo "Installing flexible version dependencies..."
	pip install -r requirements-flexible.txt
	@echo "Verifying installation..."
	python3 -c "import ludwig; print(f'Ludwig: {ludwig.__version__}')"
	python3 -c "import pydantic; print(f'Pydantic: {pydantic.__version__}')"

# 自动修复依赖冲突
fix-deps:
	@echo "Automatically fixing dependency conflicts..."
	python3 scripts/fix_dependencies.py

# 验证安装
verify:
	@echo "Verifying installation and functionality..."
	python3 scripts/verify_installation.py

# 项目设置
setup: check-deps install verify
	@echo "Setting up MLOps project..."
	python scripts/init_database.py
	python examples/generate_sample_data.py --samples 5000 --output-dir data/raw
	python examples/generate_sample_data.py --samples 2000 --generate-drift --output-dir data/raw
	@echo "Setup completed!"

# 启动服务
start: check-ports docker-up
	@echo "Starting MLOps services..."
	@echo "MLflow UI: http://localhost:5000"
	@echo "Model Registry UI: http://localhost:8080"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

# 停止服务
stop: docker-down

# 清理
clean:
	@echo "Cleaning up..."
	rm -rf data/processed/*
	rm -rf models/ludwig_output/*
	rm -rf logs/*
	rm -rf monitoring/*.db
	rm -rf state/*
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Cleanup completed!"

# 运行测试
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short

# 代码检查
lint:
	@echo "Running code linting..."
	flake8 src/ scripts/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# 代码格式化
format:
	@echo "Formatting code..."
	black src/ scripts/ tests/ examples/ --line-length=100
	isort src/ scripts/ tests/ examples/

# Docker相关命令
docker-build:
	@echo "Building Docker images..."
	cd docker && docker-compose build

docker-up:
	@echo "Starting Docker services..."
	cd docker && docker-compose up -d
	@echo "Waiting for services to start..."
	sleep 10

docker-down:
	@echo "Stopping Docker services..."
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

# 训练相关命令
train-challenger:
	@echo "Training new challenger model..."
	python scripts/train_challenger.py --data-path data/raw/baseline_full.csv --auto-evaluate

train-challenger-drift:
	@echo "Training challenger with drift data..."
	python scripts/train_challenger.py --data-path data/raw/drift_full.csv --auto-evaluate

# 运行生命周期
run-lifecycle:
	@echo "Running lifecycle cycle..."
	python scripts/run_lifecycle_cycle.py

run-lifecycle-with-data:
	@echo "Running lifecycle cycle with new data..."
	python scripts/run_lifecycle_cycle.py --data-path data/raw/drift_full.csv

# 状态检查
status:
	@echo "Checking system status..."
	python scripts/run_lifecycle_cycle.py --dry-run

# 生成数据
generate-data:
	@echo "Generating sample data..."
	python examples/generate_sample_data.py --samples 10000 --output-dir data/raw
	python examples/generate_sample_data.py --samples 5000 --generate-drift --output-dir data/raw

# 开发环境设置
dev-setup: install
	@echo "Setting up development environment..."
	pip install pytest black flake8 mypy isort pre-commit
	pre-commit install
	@echo "Development environment ready!"

# 监控相关
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "MLflow: http://localhost:5000"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# 备份和恢复
backup:
	@echo "Creating backup..."
	mkdir -p backups
	tar -czf backups/mlops-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		data/ models/ logs/ state/ config/
	@echo "Backup created in backups/"

# 性能测试
perf-test:
	@echo "Running performance tests..."
	# 这里可以添加性能测试脚本
	python -c "print('Performance testing not implemented yet')"

# 安全检查
security-check:
	@echo "Running security checks..."
	pip-audit
	bandit -r src/ -f json -o security-report.json || true
	@echo "Security check completed. Check security-report.json for details."

# 文档生成
docs:
	@echo "Generating documentation..."
	# 这里可以添加文档生成命令
	python -c "print('Documentation generation not implemented yet')"

# 完整的CI/CD流程
ci: lint test
	@echo "CI pipeline completed successfully!"

# 部署到生产环境
deploy-prod:
	@echo "Deploying to production..."
	@echo "This should be implemented based on your production environment"
	@echo "Consider using Kubernetes, AWS ECS, or other container orchestration platforms"

# 健康检查
health-check:
	@echo "Performing health check..."
	curl -f http://localhost:5000/health || echo "MLflow service not healthy"
	curl -f http://localhost:3000/api/health || echo "Grafana service not healthy"
	curl -f http://localhost:9090/-/healthy || echo "Prometheus service not healthy"

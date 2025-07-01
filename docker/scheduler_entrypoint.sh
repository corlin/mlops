#!/bin/bash

# 启动cron服务
service cron start

# 初始化数据库（如果需要）
python scripts/init_database.py

# 启动主循环，保持容器运行
echo "MLOps Scheduler started at $(date)"
echo "Running initial lifecycle cycle (simple mode)..."
python scripts/run_lifecycle_cycle_simple.py

# 保持容器运行并监控日志
tail -f /app/logs/scheduler.log &

# 等待信号
wait

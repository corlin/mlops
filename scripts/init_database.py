#!/usr/bin/env python3
"""
初始化数据库脚本
"""

import os
import sys
import sqlite3
from pathlib import Path
from loguru import logger
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def init_sqlite_database():
    """初始化SQLite数据库（用于监控）"""
    logger.info("Initializing SQLite database for monitoring")
    
    db_path = Path("monitoring/metrics.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 创建模型指标表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # 创建预测日志表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    input_data TEXT,
                    prediction TEXT,
                    latency REAL,
                    error_message TEXT
                )
            """)
            
            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp 
                ON model_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp 
                ON prediction_logs(timestamp)
            """)
            
            conn.commit()
            logger.info("SQLite database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise


def init_postgresql_database():
    """初始化PostgreSQL数据库（如果配置了的话）"""
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        logger.info("No PostgreSQL DATABASE_URL configured, skipping PostgreSQL initialization")
        return
    
    logger.info("Initializing PostgreSQL database")
    
    try:
        # 解析数据库URL
        # 格式: postgresql://user:password@host:port/database
        import urllib.parse
        parsed = urllib.parse.urlparse(database_url)
        
        # 连接到PostgreSQL服务器（不指定数据库）
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port,
            user=parsed.username,
            password=parsed.password,
            database='postgres'  # 连接到默认数据库
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        
        # 检查数据库是否存在
        database_name = parsed.path[1:]  # 移除开头的 '/'
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (database_name,))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database: {database_name}")
            cursor.execute(f'CREATE DATABASE "{database_name}"')
        
        cursor.close()
        conn.close()
        
        # 连接到目标数据库并创建表
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # 读取并执行初始化SQL
        init_sql_path = Path("docker/init.sql")
        if init_sql_path.exists():
            with open(init_sql_path, 'r') as f:
                init_sql = f.read()
            
            cursor.execute(init_sql)
            conn.commit()
            logger.info("PostgreSQL database initialized successfully")
        else:
            logger.warning("init.sql file not found, skipping table creation")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
        raise


def create_directories():
    """创建必要的目录"""
    logger.info("Creating necessary directories")
    
    directories = [
        "data/raw",
        "data/processed",
        "models/ludwig_output",
        "logs",
        "monitoring",
        "state"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def main():
    """主函数"""
    logger.info("Starting database initialization")
    
    try:
        # 创建目录
        create_directories()
        
        # 初始化SQLite数据库
        init_sqlite_database()
        
        # 初始化PostgreSQL数据库（如果配置了）
        init_postgresql_database()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

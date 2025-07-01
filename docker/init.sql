-- 初始化MLOps数据库

-- 创建模型元数据表
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'champion', 'challenger'
    status VARCHAR(50) NOT NULL, -- 'training', 'testing', 'production', 'archived'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mlflow_run_id VARCHAR(255),
    deployment_info JSONB,
    performance_metrics JSONB,
    UNIQUE(model_name, model_version)
);

-- 创建性能指标表
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- 创建影子测试记录表
CREATE TABLE IF NOT EXISTS shadow_tests (
    id SERIAL PRIMARY KEY,
    challenger_name VARCHAR(255) NOT NULL,
    champion_name VARCHAR(255) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP,
    status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed'
    test_results JSONB,
    decision VARCHAR(50), -- 'promote', 'reject'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建部署历史表
CREATE TABLE IF NOT EXISTS deployment_history (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    deployment_type VARCHAR(50) NOT NULL, -- 'champion', 'shadow'
    deployment_status VARCHAR(50) NOT NULL, -- 'deployed', 'failed', 'stopped'
    deployment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    container_info JSONB,
    error_message TEXT
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_model_metadata_name_version ON model_metadata(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(metric_timestamp);
CREATE INDEX IF NOT EXISTS idx_shadow_tests_status ON shadow_tests(status);
CREATE INDEX IF NOT EXISTS idx_deployment_history_time ON deployment_history(deployment_time);

-- 插入初始数据（如果需要）
INSERT INTO model_metadata (model_name, model_version, model_type, status) 
VALUES ('initial_model', '1.0.0', 'champion', 'production')
ON CONFLICT (model_name, model_version) DO NOTHING;

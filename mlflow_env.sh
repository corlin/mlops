
# MLflow环境配置
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlflow_artifacts
export MLFLOW_BACKEND_STORE_URI=./mlruns

# 确保目录存在
mkdir -p mlflow_artifacts
mkdir -p mlruns
mkdir -p logs

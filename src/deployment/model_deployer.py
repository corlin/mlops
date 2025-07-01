"""
模型部署模块 - 负责容器化部署和服务管理
"""

import yaml
import docker
import json
import requests
from typing import Dict, List, Any, Optional
from loguru import logger
from pathlib import Path
import tempfile
import shutil
import time
from datetime import datetime


class ModelDeployer:
    """模型部署器，负责容器化部署和服务管理"""
    
    def __init__(self, config_path: str):
        """
        初始化部署器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.deployment_config = self.config['deployment']

        # 初始化Docker客户端
        self._init_docker_client()

    def _init_docker_client(self):
        """初始化Docker客户端"""
        try:
            self.docker_client = docker.from_env()
            # 测试Docker连接
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
            logger.warning("Docker deployment features will be disabled")
            self.docker_client = None

    def _check_docker_available(self):
        """检查Docker是否可用"""
        if self.docker_client is None:
            logger.warning("Docker client not available. Skipping Docker operation.")
            return False
        return True
    
    def create_model_service_dockerfile(self, model_name: str, model_version: str) -> str:
        """
        创建模型服务的Dockerfile
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            Dockerfile内容
        """
        dockerfile_content = f"""
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV MODEL_NAME={model_name}
ENV MODEL_VERSION={model_version}
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "model_service.py"]
"""
        return dockerfile_content.strip()
    
    def create_model_service_code(self, model_name: str, model_version: str) -> str:
        """
        创建模型服务代码
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            服务代码
        """
        service_code = f"""
import os
import json
import mlflow
try:
    import mlflow.ludwig
    MLFLOW_LUDWIG_AVAILABLE = True
except ImportError:
    MLFLOW_LUDWIG_AVAILABLE = False
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import uvicorn
from loguru import logger
import time
from datetime import datetime

# 初始化FastAPI应用
app = FastAPI(title="Model Service", version="1.0.0")

# 模型配置
MODEL_NAME = os.getenv("MODEL_NAME", "{model_name}")
MODEL_VERSION = os.getenv("MODEL_VERSION", "{model_version}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# 全局变量
model = None
model_metadata = {{}}

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_name: str
    model_version: str
    timestamp: str

@app.on_event("startup")
async def load_model():
    global model, model_metadata
    
    try:
        logger.info(f"Loading model {{MODEL_NAME}} version {{MODEL_VERSION}}")
        
        # 设置MLflow跟踪URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # 加载模型
        model_uri = f"models:/{{MODEL_NAME}}/{{MODEL_VERSION}}"
        try:
            if MLFLOW_LUDWIG_AVAILABLE:
                model = mlflow.ludwig.load_model(model_uri)
            else:
                # 使用通用MLflow加载方法
                model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {{e}}")
            # 最后尝试使用通用方法
            model = mlflow.pyfunc.load_model(model_uri)
        
        # 获取模型元数据
        client = mlflow.tracking.MlflowClient()
        model_version_info = client.get_model_version(MODEL_NAME, MODEL_VERSION)
        
        model_metadata = {{
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "creation_timestamp": model_version_info.creation_timestamp,
            "last_updated_timestamp": model_version_info.last_updated_timestamp,
            "stage": model_version_info.current_stage,
            "run_id": model_version_info.run_id
        }}
        
        logger.info(f"Model loaded successfully: {{MODEL_NAME}} v{{MODEL_VERSION}}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise

@app.get("/health")
async def health_check():
    return {{
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }}

@app.get("/model/info")
async def get_model_info():
    return model_metadata

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # 转换输入数据
        input_df = pd.DataFrame(request.data)
        
        # 进行预测
        predictions = model.predict(input_df)
        
        # 转换预测结果
        prediction_list = []
        for idx, row in predictions.iterrows():
            prediction_list.append(row.to_dict())
        
        inference_time = time.time() - start_time
        
        # 记录预测日志
        logger.info(f"Prediction completed - samples: {{len(request.data)}}, time: {{inference_time:.3f}}s")
        
        return PredictionResponse(
            predictions=prediction_list,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{str(e)}}")

@app.post("/batch_predict")
async def batch_predict(request: PredictionRequest):
    # 批量预测的实现
    return await predict(request)

if __name__ == "__main__":
    uvicorn.run(
        "model_service:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
"""
        return service_code.strip()
    
    def build_model_image(self, model_name: str, model_version: str, image_tag: str = None) -> str:
        """
        构建模型Docker镜像
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            image_tag: 镜像标签
            
        Returns:
            镜像名称
        """
        if image_tag is None:
            image_tag = f"{model_name}:{model_version}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        try:
            # 创建临时构建目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # 创建Dockerfile
                dockerfile_content = self.create_model_service_dockerfile(model_name, model_version)
                with open(temp_path / "Dockerfile", 'w') as f:
                    f.write(dockerfile_content)
                
                # 创建服务代码
                service_code = self.create_model_service_code(model_name, model_version)
                with open(temp_path / "model_service.py", 'w') as f:
                    f.write(service_code)
                
                # 复制requirements.txt
                shutil.copy("requirements.txt", temp_path / "requirements.txt")
                
                # 构建镜像
                image, build_logs = self.docker_client.images.build(
                    path=str(temp_path),
                    tag=image_tag,
                    rm=True,
                    forcerm=True
                )
                
                # 记录构建日志
                for log in build_logs:
                    if 'stream' in log:
                        logger.debug(log['stream'].strip())
                
                logger.info(f"Docker image built successfully: {image_tag}")
                return image_tag
                
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise
    
    def deploy_champion(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        部署冠军模型

        Args:
            model_name: 模型名称
            model_version: 模型版本

        Returns:
            部署结果
        """
        logger.info(f"Deploying champion model: {model_name} v{model_version}")

        if not self._check_docker_available():
            logger.warning("Skipping champion deployment due to Docker unavailability")
            return False

        try:
            # 构建镜像
            image_tag = f"{model_name}-champion:{model_version}"
            self.build_model_image(model_name, model_version, image_tag)
            
            # 停止现有的冠军容器
            try:
                existing_container = self.docker_client.containers.get("champion-model")
                existing_container.stop()
                existing_container.remove()
                logger.info("Stopped existing champion container")
            except docker.errors.NotFound:
                logger.info("No existing champion container found")
            
            # 启动新的冠军容器
            container = self.docker_client.containers.run(
                image_tag,
                name="champion-model",
                ports={'8000/tcp': 8000},
                environment={
                    'MODEL_NAME': model_name,
                    'MODEL_VERSION': model_version,
                    'MLFLOW_TRACKING_URI': self.config['mlflow']['tracking_uri']
                },
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            # 等待容器启动
            time.sleep(10)
            
            # 健康检查
            health_check_passed = self._wait_for_health_check("http://localhost:8000/health")
            
            deployment_result = {
                'container_id': container.id,
                'container_name': container.name,
                'image_tag': image_tag,
                'port': 8000,
                'health_check_passed': health_check_passed,
                'deployment_time': datetime.now().isoformat()
            }
            
            logger.info(f"Champion model deployed successfully: {model_name}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Champion deployment failed: {e}")
            raise
    
    def deploy_shadow(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        部署影子模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            
        Returns:
            部署结果
        """
        logger.info(f"Deploying shadow model: {model_name} v{model_version}")
        
        try:
            # 构建镜像
            image_tag = f"{model_name}-shadow:{model_version}"
            self.build_model_image(model_name, model_version, image_tag)
            
            # 启动影子容器
            container_name = f"shadow-{model_name}"
            container = self.docker_client.containers.run(
                image_tag,
                name=container_name,
                ports={'8000/tcp': 8001},  # 使用不同端口
                environment={
                    'MODEL_NAME': model_name,
                    'MODEL_VERSION': model_version,
                    'MLFLOW_TRACKING_URI': self.config['mlflow']['tracking_uri']
                },
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            # 等待容器启动
            time.sleep(10)
            
            # 健康检查
            health_check_passed = self._wait_for_health_check("http://localhost:8001/health")
            
            deployment_result = {
                'container_id': container.id,
                'container_name': container.name,
                'image_tag': image_tag,
                'port': 8001,
                'health_check_passed': health_check_passed,
                'deployment_time': datetime.now().isoformat()
            }
            
            logger.info(f"Shadow model deployed successfully: {model_name}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Shadow deployment failed: {e}")
            raise
    
    def cleanup_shadow(self, model_name: str) -> None:
        """
        清理影子部署
        
        Args:
            model_name: 模型名称
        """
        logger.info(f"Cleaning up shadow deployment: {model_name}")
        
        try:
            container_name = f"shadow-{model_name}"
            container = self.docker_client.containers.get(container_name)
            container.stop()
            container.remove()
            logger.info(f"Shadow container removed: {container_name}")
            
        except docker.errors.NotFound:
            logger.info(f"Shadow container not found: {model_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup shadow deployment: {e}")
            raise
    
    def _wait_for_health_check(self, health_url: str, max_attempts: int = 30, delay: int = 2) -> bool:
        """
        等待健康检查通过
        
        Args:
            health_url: 健康检查URL
            max_attempts: 最大尝试次数
            delay: 延迟时间
            
        Returns:
            是否通过健康检查
        """
        logger.info(f"Waiting for health check: {health_url}")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        logger.info("Health check passed")
                        return True
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
            
            time.sleep(delay)
        
        logger.warning("Health check failed after maximum attempts")
        return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """
        获取部署状态
        
        Returns:
            部署状态信息
        """
        status = {
            'champion': None,
            'shadows': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 检查冠军容器
            try:
                champion_container = self.docker_client.containers.get("champion-model")
                status['champion'] = {
                    'container_id': champion_container.id,
                    'status': champion_container.status,
                    'image': champion_container.image.tags[0] if champion_container.image.tags else None,
                    'ports': champion_container.ports
                }
            except docker.errors.NotFound:
                pass
            
            # 检查影子容器
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                if container.name.startswith("shadow-"):
                    status['shadows'].append({
                        'container_id': container.id,
                        'name': container.name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else None,
                        'ports': container.ports
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return status

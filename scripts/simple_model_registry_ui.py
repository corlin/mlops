#!/usr/bin/env python3
"""
简化版模型注册表Web UI - 避免复杂依赖
"""

import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
from datetime import datetime
import requests

# 页面配置
st.set_page_config(
    page_title="MLOps Model Registry",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化MLflow
@st.cache_resource
def init_mlflow():
    """初始化MLflow客户端"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    return MlflowClient()

def get_registered_models(client):
    """获取注册的模型列表"""
    try:
        models = client.search_registered_models()
        return models
    except Exception as e:
        st.error(f"Failed to fetch registered models: {e}")
        return []

def get_model_versions(client, model_name):
    """获取模型版本"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return sorted(versions, key=lambda x: int(x.version), reverse=True)
    except Exception as e:
        st.error(f"Failed to fetch model versions: {e}")
        return []

def get_experiment_runs(experiment_name):
    """获取实验运行记录"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch experiment runs: {e}")
        return pd.DataFrame()

def check_mlflow_connection():
    """检查MLflow连接"""
    try:
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        response = requests.get(f"{mlflow_uri}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def main():
    """主函数"""
    st.title("🤖 MLOps Model Registry Dashboard")
    st.markdown("---")
    
    # 检查MLflow连接
    if not check_mlflow_connection():
        st.error("❌ Cannot connect to MLflow server. Please check if MLflow is running.")
        st.info("MLflow should be available at: http://mlflow:5000")
        return
    
    # 初始化客户端
    try:
        client = init_mlflow()
        st.success("✅ Connected to MLflow")
    except Exception as e:
        st.error(f"❌ Failed to initialize MLflow client: {e}")
        return
    
    # 侧边栏
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Registry", "Experiments"]
    )
    
    if page == "Overview":
        show_overview(client)
    elif page == "Model Registry":
        show_model_registry(client)
    elif page == "Experiments":
        show_experiments()

def show_overview(client):
    """显示概览页面"""
    st.header("📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # 统计信息
    try:
        models = get_registered_models(client)
        total_models = len(models)
        
        production_models = 0
        staging_models = 0
        
        for model in models:
            versions = get_model_versions(client, model.name)
            for version in versions:
                if version.current_stage == "Production":
                    production_models += 1
                elif version.current_stage == "Staging":
                    staging_models += 1
        
        with col1:
            st.metric("Total Models", total_models)
        
        with col2:
            st.metric("Production Models", production_models)
        
        with col3:
            st.metric("Staging Models", staging_models)
        
        # 显示模型列表
        if models:
            st.subheader("📋 Registered Models")
            model_data = []
            for model in models:
                versions = get_model_versions(client, model.name)
                latest_version = versions[0] if versions else None
                
                model_data.append({
                    "Name": model.name,
                    "Latest Version": latest_version.version if latest_version else "N/A",
                    "Stage": latest_version.current_stage if latest_version else "N/A",
                    "Created": datetime.fromtimestamp(model.creation_timestamp / 1000).strftime("%Y-%m-%d") if model.creation_timestamp else "N/A"
                })
            
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No registered models found")
        
    except Exception as e:
        st.error(f"Failed to load overview data: {e}")

def show_model_registry(client):
    """显示模型注册表页面"""
    st.header("📋 Model Registry")
    
    models = get_registered_models(client)
    
    if not models:
        st.info("No registered models found")
        return
    
    # 模型选择
    model_names = [model.name for model in models]
    selected_model = st.selectbox("Select a model", model_names)
    
    if selected_model:
        st.subheader(f"Model: {selected_model}")
        
        # 获取模型版本
        versions = get_model_versions(client, selected_model)
        
        if versions:
            # 版本表格
            version_data = []
            for version in versions:
                version_data.append({
                    "Version": version.version,
                    "Stage": version.current_stage,
                    "Created": datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "Run ID": version.run_id[:8] + "...",
                    "Description": version.description or "No description"
                })
            
            df = pd.DataFrame(version_data)
            st.dataframe(df, use_container_width=True)
            
            # 版本详情
            selected_version = st.selectbox("Select version for details", [v.version for v in versions])
            
            if selected_version:
                version_detail = next(v for v in versions if v.version == selected_version)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Version Information:**")
                    st.write(f"Version: {version_detail.version}")
                    st.write(f"Stage: {version_detail.current_stage}")
                    st.write(f"Run ID: {version_detail.run_id}")
                    st.write(f"Created: {datetime.fromtimestamp(version_detail.creation_timestamp / 1000)}")
                
                with col2:
                    st.write("**Tags:**")
                    if version_detail.tags:
                        for key, value in version_detail.tags.items():
                            st.write(f"{key}: {value}")
                    else:
                        st.write("No tags")
                
                # 模型阶段转换
                st.subheader("Stage Transition")
                new_stage = st.selectbox(
                    "Transition to stage",
                    ["None", "Staging", "Production", "Archived"],
                    index=0
                )
                
                if st.button("Transition Stage") and new_stage != "None":
                    try:
                        client.transition_model_version_stage(
                            name=selected_model,
                            version=selected_version,
                            stage=new_stage
                        )
                        st.success(f"Model transitioned to {new_stage}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to transition stage: {e}")

def show_experiments():
    """显示实验页面"""
    st.header("🧪 Experiments")
    
    # 获取实验列表
    try:
        client = init_mlflow()
        experiments = client.search_experiments()
        
        if not experiments:
            st.info("No experiments found")
            return
        
        # 实验选择
        exp_names = [exp.name for exp in experiments if exp.name != "Default"]
        if not exp_names:
            st.info("No named experiments found")
            return
        
        selected_exp = st.selectbox("Select experiment", exp_names)
        
        if selected_exp:
            # 获取运行记录
            runs_df = get_experiment_runs(selected_exp)
            
            if not runs_df.empty:
                st.subheader(f"Runs in {selected_exp}")
                
                # 显示运行统计
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Runs", len(runs_df))
                
                with col2:
                    finished_runs = len(runs_df[runs_df['status'] == 'FINISHED'])
                    st.metric("Finished Runs", finished_runs)
                
                with col3:
                    failed_runs = len(runs_df[runs_df['status'] == 'FAILED'])
                    st.metric("Failed Runs", failed_runs)
                
                # 运行表格
                display_columns = ['run_id', 'status', 'start_time', 'end_time']
                metric_columns = [col for col in runs_df.columns if col.startswith('metrics.')]
                display_columns.extend(metric_columns[:5])  # 显示前5个指标
                
                if all(col in runs_df.columns for col in display_columns):
                    st.dataframe(runs_df[display_columns], use_container_width=True)
            
            else:
                st.info(f"No runs found in experiment {selected_exp}")
    
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
模型注册表Web UI - 使用Streamlit构建
"""

import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lifecycle import ChampionChallengerManager


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
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    return MlflowClient()

@st.cache_resource
def init_cc_manager():
    """初始化冠军挑战者管理器"""
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        return ChampionChallengerManager(config_path)
    return None

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

def main():
    """主函数"""
    st.title("🤖 MLOps Model Registry Dashboard")
    st.markdown("---")
    
    # 初始化客户端
    client = init_mlflow()
    cc_manager = init_cc_manager()
    
    # 侧边栏
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Model Registry", "Experiments", "Champion-Challenger", "Monitoring"]
    )
    
    if page == "Overview":
        show_overview(client, cc_manager)
    elif page == "Model Registry":
        show_model_registry(client)
    elif page == "Experiments":
        show_experiments()
    elif page == "Champion-Challenger":
        show_champion_challenger(cc_manager)
    elif page == "Monitoring":
        show_monitoring(cc_manager)

def show_overview(client, cc_manager):
    """显示概览页面"""
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
        
        with col4:
            if cc_manager:
                status = cc_manager.get_status()
                active_tests = len(status.get('active_shadow_tests', []))
                st.metric("Active Shadow Tests", active_tests)
            else:
                st.metric("Active Shadow Tests", "N/A")
        
    except Exception as e:
        st.error(f"Failed to load overview data: {e}")
    
    # 系统状态
    if cc_manager:
        st.subheader("🎯 Champion-Challenger Status")
        status = cc_manager.get_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Champion:**")
            champion = status.get('champion_model')
            if champion:
                st.json(champion)
            else:
                st.info("No champion model deployed")
        
        with col2:
            st.write("**Active Challengers:**")
            challengers = status.get('challenger_models', [])
            if challengers:
                for challenger in challengers:
                    st.json(challenger)
            else:
                st.info("No active challengers")

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
                
                # 指标可视化
                if metric_columns:
                    st.subheader("Metrics Visualization")
                    
                    selected_metric = st.selectbox("Select metric to visualize", metric_columns)
                    
                    if selected_metric:
                        fig = px.line(
                            runs_df, 
                            x='start_time', 
                            y=selected_metric,
                            title=f"{selected_metric} over time"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info(f"No runs found in experiment {selected_exp}")
    
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")

def show_champion_challenger(cc_manager):
    """显示冠军挑战者页面"""
    st.header("🏆 Champion-Challenger Management")
    
    if not cc_manager:
        st.error("Champion-Challenger Manager not available. Check configuration.")
        return
    
    # 获取状态
    status = cc_manager.get_status()
    
    # 当前状态
    st.subheader("Current Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Champion Model:**")
        champion = status.get('champion_model')
        if champion:
            st.json(champion)
        else:
            st.info("No champion model")
    
    with col2:
        st.write("**System Health:**")
        health = status.get('system_health', {})
        if health.get('overall_status') == 'healthy':
            st.success("System is healthy")
        else:
            st.warning("System has issues")
        
        if health.get('alerts'):
            for alert in health['alerts']:
                st.warning(alert)
    
    # 挑战者模型
    st.subheader("Challenger Models")
    challengers = status.get('challenger_models', [])
    
    if challengers:
        for challenger in challengers:
            with st.expander(f"Challenger: {challenger['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(challenger)
                
                with col2:
                    if st.button(f"Evaluate {challenger['name']}", key=f"eval_{challenger['name']}"):
                        try:
                            with st.spinner("Evaluating challenger..."):
                                result = cc_manager.evaluate_challenger(challenger['name'])
                            st.success(f"Evaluation completed: {result.get('action', 'unknown')}")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
    else:
        st.info("No challenger models")
    
    # 影子测试
    st.subheader("Shadow Tests")
    shadow_tests = status.get('active_shadow_tests', [])
    
    if shadow_tests:
        for test in shadow_tests:
            with st.expander(f"Shadow Test: {test['challenger_name']}"):
                st.json(test)
    else:
        st.info("No active shadow tests")
    
    # 手动操作
    st.subheader("Manual Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Lifecycle Cycle"):
            try:
                with st.spinner("Running lifecycle cycle..."):
                    result = cc_manager.run_lifecycle_cycle()
                st.success("Lifecycle cycle completed")
                st.json(result)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Lifecycle cycle failed: {e}")
    
    with col2:
        if st.button("Monitor Shadow Tests"):
            try:
                with st.spinner("Monitoring shadow tests..."):
                    results = cc_manager.monitor_shadow_tests()
                st.success("Shadow test monitoring completed")
                if results:
                    for result in results:
                        st.json(result)
                else:
                    st.info("No shadow tests to monitor")
            except Exception as e:
                st.error(f"Shadow test monitoring failed: {e}")

def show_monitoring(cc_manager):
    """显示监控页面"""
    st.header("📈 Model Monitoring")
    
    if not cc_manager:
        st.error("Monitoring not available. Check configuration.")
        return
    
    # 系统健康状态
    st.subheader("System Health")
    
    try:
        health = cc_manager.model_monitor.get_system_health()
        
        if health['overall_status'] == 'healthy':
            st.success("✅ System is healthy")
        else:
            st.warning("⚠️ System has issues")
        
        # 组件状态
        st.write("**Component Status:**")
        for component, status in health.get('components', {}).items():
            if status == 'healthy':
                st.success(f"{component}: {status}")
            else:
                st.error(f"{component}: {status}")
        
        # 告警
        alerts = health.get('alerts', [])
        if alerts:
            st.write("**Active Alerts:**")
            for alert in alerts:
                st.warning(alert)
    
    except Exception as e:
        st.error(f"Failed to get system health: {e}")
    
    # 部署状态
    st.subheader("Deployment Status")
    
    try:
        deployment_status = cc_manager.model_deployer.get_deployment_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Champion Deployment:**")
            champion_deploy = deployment_status.get('champion')
            if champion_deploy:
                st.json(champion_deploy)
            else:
                st.info("No champion deployed")
        
        with col2:
            st.write("**Shadow Deployments:**")
            shadow_deploys = deployment_status.get('shadows', [])
            if shadow_deploys:
                for shadow in shadow_deploys:
                    st.json(shadow)
            else:
                st.info("No shadow deployments")
    
    except Exception as e:
        st.error(f"Failed to get deployment status: {e}")


if __name__ == "__main__":
    main()

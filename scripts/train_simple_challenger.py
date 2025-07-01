#!/usr/bin/env python3
"""
简化版挑战者模型训练脚本 - 避免复杂依赖
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import pickle
import os
from datetime import datetime
from pathlib import Path
import json

def setup_mlflow():
    """设置MLflow"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow.set_tracking_uri(mlflow_uri)
    
    experiment_name = "simple_champion_challenger"
    try:
        # 尝试获取现有实验
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage == 'deleted':
            # 如果实验被删除，创建新的实验名称
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"simple_champion_challenger_{timestamp}"
            experiment_id = mlflow.create_experiment(experiment_name)
        elif experiment:
            experiment_id = experiment.experiment_id
        else:
            # 实验不存在，创建新的
            experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        # 如果所有方法都失败，使用默认实验
        experiment_id = "0"
        experiment_name = "Default"

    mlflow.set_experiment(experiment_name)
    return experiment_id

def load_and_prepare_data(data_path):
    """加载和准备数据"""
    print(f"Loading data from: {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 检查目标列
    if 'target' not in df.columns:
        print("Warning: 'target' column not found. Using last column as target.")
        target_col = df.columns[-1]
        df = df.rename(columns={target_col: 'target'})
    
    # 分离特征和目标
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 处理分类特征
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_model(X, y, model_type='random_forest'):
    """训练模型"""
    print(f"Training {model_type} model...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 选择模型
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    print("Model performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, metrics, X_test, y_test

def log_to_mlflow(model, metrics, model_type, model_name):
    """记录到MLflow"""
    print("Logging to MLflow...")
    
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # 记录参数
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        if hasattr(model, 'get_params'):
            for param, value in model.get_params().items():
                mlflow.log_param(f"model_{param}", value)
        
        # 记录指标
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # 保存模型到本地（避免MLflow版本兼容问题）
        import pickle
        import tempfile

        model_dir = Path("models") / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 记录模型路径（不上传artifact）
        mlflow.log_param("model_path", str(model_path))
        print(f"Model saved locally: {model_path}")
        
        # 记录模型信息到文件
        model_info = {
            'model_name': model_name,
            'model_type': model_type,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'run_id': mlflow.active_run().info.run_id
        }
        
        # 保存到本地状态文件
        state_dir = Path("state")
        state_dir.mkdir(exist_ok=True)
        
        with open(state_dir / f"{model_name}_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model logged with run ID: {mlflow.active_run().info.run_id}")
        return mlflow.active_run().info.run_id

def evaluate_against_champion(metrics, model_name):
    """简单的冠军挑战者评估"""
    print("\nEvaluating against champion...")
    
    state_dir = Path("state")
    champion_file = state_dir / "champion_model.json"
    
    if not champion_file.exists():
        print("No champion model found. This model becomes the champion!")
        
        champion_info = {
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'status': 'champion'
        }
        
        with open(champion_file, 'w') as f:
            json.dump(champion_info, f, indent=2)
        
        return "promoted_to_champion"
    
    # 加载当前冠军
    with open(champion_file, 'r') as f:
        champion_info = json.load(f)
    
    champion_accuracy = champion_info['metrics']['accuracy']
    challenger_accuracy = metrics['accuracy']
    
    improvement = challenger_accuracy - champion_accuracy
    threshold = 0.01  # 1% improvement threshold
    
    print(f"Champion accuracy: {champion_accuracy:.4f}")
    print(f"Challenger accuracy: {challenger_accuracy:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    if improvement > threshold:
        print("🎉 Challenger performs better! Promoting to champion.")
        
        # 备份旧冠军
        backup_file = state_dir / f"champion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, 'w') as f:
            json.dump(champion_info, f, indent=2)
        
        # 更新冠军
        new_champion_info = {
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'status': 'champion',
            'previous_champion': champion_info['model_name']
        }
        
        with open(champion_file, 'w') as f:
            json.dump(new_champion_info, f, indent=2)
        
        return "promoted_to_champion"
    else:
        print("Challenger does not meet improvement threshold. Champion remains unchanged.")
        return "challenger_rejected"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train a simple challenger model")
    parser.add_argument("--data-path", required=True, help="Path to training data CSV")
    parser.add_argument("--model-type", default="random_forest", 
                       choices=["random_forest", "logistic_regression"],
                       help="Type of model to train")
    parser.add_argument("--model-name", help="Name for the model")
    parser.add_argument("--auto-evaluate", action="store_true", 
                       help="Automatically evaluate against champion")
    
    args = parser.parse_args()
    
    # 生成模型名称
    if not args.model_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"{args.model_type}_challenger_{timestamp}"
    
    print("🚀 Starting Simple Challenger Training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Model name: {args.model_name}")
    print(f"Auto evaluate: {args.auto_evaluate}")
    print("=" * 50)
    
    try:
        # 设置MLflow
        experiment_id = setup_mlflow()
        print(f"MLflow experiment ID: {experiment_id}")
        
        # 加载数据
        X, y = load_and_prepare_data(args.data_path)
        
        # 训练模型
        model, metrics, X_test, y_test = train_model(X, y, args.model_type)
        
        # 记录到MLflow
        run_id = log_to_mlflow(model, metrics, args.model_type, args.model_name)
        
        # 自动评估
        if args.auto_evaluate:
            result = evaluate_against_champion(metrics, args.model_name)
            print(f"\nEvaluation result: {result}")
        
        print("\n🎉 Training completed successfully!")
        print(f"Model: {args.model_name}")
        print(f"Run ID: {run_id}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # 显示访问信息
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        print(f"\n📊 View results in MLflow: {mlflow_uri}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

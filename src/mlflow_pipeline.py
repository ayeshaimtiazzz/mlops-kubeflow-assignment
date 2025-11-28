# src/mlflow_pipeline.py

import mlflow
from pipeline_components import extract_data, preprocess_data, train_model, evaluate_model
import mlflow.sklearn

mlflow.set_tracking_uri("file:mlruns")  # Use a folder in current directory
mlflow.set_experiment("MLOps Assignment Pipeline")

with mlflow.start_run(run_name="Full_MLOps_Pipeline"):
    mlflow.log_param("example_param", 123)
    mlflow.log_metric("example_metric", 0.95)
def run_pipeline():
    with mlflow.start_run(run_name="Full_MLOps_Pipeline"):

        with mlflow.start_run(run_name="Data_Extraction", nested=True):
            df = extract_data()

        with mlflow.start_run(run_name="Preprocessing", nested=True):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
            mlflow.sklearn.log_model(scaler, "scaler")

        with mlflow.start_run(run_name="Training", nested=True):
            model = train_model(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")

        with mlflow.start_run(run_name="Evaluation", nested=True):
            metrics = evaluate_model(model, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

if __name__ == "__main__":
    run_pipeline()

# src/pipeline_components.py

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# -------------------------
# 1. DATA EXTRACTION
# -------------------------
def extract_data(data_path=None):
    print("Extracting dataset...")
    if data_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(project_root, "data", "raw_data.csv")
    
    df = pd.read_csv(data_path)

    # Log dataset info
    mlflow.log_param("dataset_path", data_path)
    mlflow.log_param("num_rows", df.shape[0])
    mlflow.log_param("num_columns", df.shape[1])

    # Optionally log a small sample as artifact
    sample_path = os.path.join("tmp_sample.csv")
    df.head(10).to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path, artifact_path="data_samples")

    return df

# -------------------------
# 2. DATA PREPROCESSING
# -------------------------
def preprocess_data(df, target_column="MEDV"):
    print("Preprocessing data...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Log preprocessing info
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("train_test_split", "80/20")
    mlflow.log_param("num_features", X.shape[1])
    mlflow.sklearn.log_model(scaler, "scaler")

    return X_train, X_test, y_train, y_test, scaler

# -------------------------
# 3. MODEL TRAINING
# -------------------------
def train_model(X_train, y_train, n_estimators=100):
    print("Training Random Forest Regressor model...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Log model and hyperparameters
    mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("num_training_samples", X_train.shape[0])
    mlflow.log_param("num_features", X_train.shape[1])

    return model

# -------------------------
# 4. MODEL EVALUATION
# -------------------------
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    # Log metrics
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    return {"MSE": mse, "R2": r2}

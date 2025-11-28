# MLOps Pipeline with MLflow

## Project Overview
This project demonstrates a complete **MLOps workflow** for a machine learning pipeline using **MLflow** for experiment tracking and reproducibility.  

The pipeline includes the following steps:

1. **Data Extraction**: Loading raw data from CSV files.
2. **Data Preprocessing**: Scaling features and splitting the dataset into training and test sets.
3. **Model Training**: Training a Random Forest Regressor.
4. **Evaluation**: Computing metrics such as Mean Squared Error (MSE) and R² score.
5. **Logging & Tracking**: MLflow logs parameters, metrics, and models for each pipeline step.

The project showcases **end-to-end pipeline automation** and **tracking experiments** in MLflow.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <YOUR_REPO_URL>
cd mlops-kubeflow-assignment

2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
### 2. MLflow Tracking

Run MLflow UI to monitor your pipeline experiments:

mlflow ui

Open the UI at: http://localhost:5000

MLflow stores all experiment runs locally in the mlruns/ folder.

### 3.Pipeline Walkthrough
1. Run the Pipeline

The pipeline is defined in src/mlflow_pipeline.py using your modular components in pipeline_components.py.
```bash
python src/mlflow_pipeline.py
```

Steps executed:

Data Extraction: extract_data() loads the dataset.

Preprocessing: preprocess_data() scales and splits the data.

Training: train_model() trains the Random Forest model.

Evaluation: evaluate_model() computes metrics and logs them in MLflow.

### 4.Check Experiments in MLflow

Each pipeline run is logged as an MLflow run under the experiment MLOps Assignment Pipeline.

Metrics like MSE and R² can be viewed under the Evaluation run.

Models and preprocessors (scaler) are logged and can be downloaded for future use.

### 5.Continuous Integration

A GitHub Actions workflow is set up to:

Checkout the code.
Install dependencies from requirements.txt.
Execute the MLflow pipeline script.
Ensure that all stages run successfully.

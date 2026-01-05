# MLOps Assignment 1 - Group 90

## Overview
This project demonstrates a full MLOps workflow for developing, training, and deploying a heart disease prediction model using the UCI dataset. The process includes data acquisition, exploratory data analysis (EDA), model development, experiment tracking, CI/CD pipelines, Dockerization, Kubernetes deployment, and monitoring.

## 1. Project Architecture
A complete end-to-end architecture has been followed involving data preprocessing, ML model training with MLflow tracking, API development using FastAPI, containerization with Docker, deployment on Kubernetes (via Minikube), and optional monitoring using Prometheus and Grafana.

## 2. Setup & Installation Instructions

### 2.1 Prerequisites
- Python 3.10
- Conda or venv
- Docker
- Minikube + kubectl
- MLflow
- GitHub account

### 2.2 Local Environment Setup (Laptop)
```bash
conda create -n mlops-a1-group90 python=3.10 -y
conda activate mlops-a1-group90
pip install -r requirements.txt
pytest -q
python -m src.train
```

### 2.3 Run FastAPI Locally
```bash
uvicorn api.app:app --reload
```

### 2.4 Docker (Local)
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```

### 2.5 AWS (OSHA Environment)
- Same steps repeated in cloud terminal.
- Additional: `minikube` setup, Kubernetes manifests in `k8s/` applied.

## 3. Data Acquisition & EDA
- Download script: `scripts/download_data.py`
- Dataset used: UCI Heart Disease Dataset
- Cleaned data stored in `data/processed/uci_heart_extracted`
- EDA done in `notebooks/eda.ipynb`
  - Histogram plots
  - Correlation heatmap
  - Class imbalance visualization

## 4. Feature Engineering & Model Development
- Final features scaled and encoded using `ColumnTransformer`
- Two models trained: Logistic Regression and Random Forest
- Model training and evaluation handled in `src/train.py`
- Evaluation metrics: Accuracy, Precision, Recall, ROC-AUC
- Confusion matrix and ROC plotted and saved as artifacts

## 5. Experiment Tracking
- MLflow integrated in training script
- Parameters, metrics, artifacts, and models logged
- MLflow UI screenshots included in `screenshots/mlflow.png`

## 6. Model Packaging & Reproducibility
- Final model saved as `artifacts/model.joblib`
- MLflow also logs models with `log_model`
- `requirements.txt` ensures reproducibility

## 7. CI/CD Pipeline & Automated Testing
- GitHub Actions configured via `.github/workflows/main.yml`
- Includes linting (flake8), testing (pytest), and model training
- All tests placed in `tests/`
- Screenshot of CI pass included

## 8. Model Containerization
- API in `api/app.py` (FastAPI)
- Dockerized with `Dockerfile`
- `/predict` and `/health` endpoints exposed
- Sample input tested via `curl`

## 9. Production Deployment
- Kubernetes manifests: `k8s/deployment.yaml`, `service.yaml`, `ingress.yaml`
- Deployed to Minikube in OSHA cloud VM
- API exposed via Ingress at `heart.local`
- Screenshots included: Swagger UI, kubectl outputs

## 10. Monitoring & Logging
- Logging handled via Python `logging` module in API
- Optionally integrated `prometheus_fastapi_instrumentator`
- Attempted integration with Prometheus + Grafana
- Screenshots included in `screenshots/`

## 11. Documentation & Reporting
- This README serves as Markdown report
- Separate `.docx` and `.pdf` reports also available
- Folder structure shared using `tree /f > project_structure.txt`
- Architecture diagram and all screenshots included

## 12. Deliverables
- GitHub Repo with code, Dockerfile, requirements.txt
- Cleaned data and download script
- Jupyter notebooks for EDA and evaluation
- Unit tests in `tests/`
- GitHub Actions workflow YAML
- Kubernetes manifests (Helm not used)
- Screenshot folder
- Written report (docx/pdf)
- End-to-end demo video (**from Google Drive**): https://drive.google.com/file/d/1jVhmXFbf5mcKs6W3Q9vYiYu625gHQzKQ/view?usp=sharing
- Instead of Public API URL, local Minikube used

## Repository Link
[GitHub Repo](https://github.com/ranga-srinivasan/mlops-assignment1-group90.git)

---
**Group:** 90

**Note:** This README omits screenshots and visuals. Refer to full `.docx` or `.pdf` report for embedded images and figures.


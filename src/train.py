"""
Training script:
- Loads dataset from ZIP
- Builds preprocessing pipeline
- Trains at least two classification models (LogReg and RandomForest)
- Evaluates with metrics (accuracy, precision, recall, roc_auc)
- Logs everything to MLflow: params, metrics, artifacts, and the trained model

This directly satisfies the assignment requirements for:
  - model development + evaluation
  - experiment tracking with MLflow
  - model packaging for reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data_processing import (
    DatasetPaths,
    load_processed_dataset,
    build_preprocess_pipeline,
    split_data,
)


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Computes standard binary classification metrics.
    y_prob: probability of class 1
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn_fp_fn_tp": [int(x) for x in confusion_matrix(y_true, y_pred).ravel()],
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    zip_path = project_root / "data" / "raw" / "heart+disease.zip"
    extracted_dir = project_root / "data" / "processed" / "uci_heart_extracted"

    paths = DatasetPaths(zip_path=zip_path, processed_dir=extracted_dir)

    # Load one dataset (recommended: cleveland for standard baseline)
    df = load_processed_dataset(paths, which="cleveland")

    train_df, val_df, test_df = split_data(df)

    preprocess, numeric_cols, categorical_cols = build_preprocess_pipeline(df)

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"].values
    X_val, y_val = val_df.drop(columns=["target"]), val_df["target"].values
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"].values

    # Define models required by the assignment (at least 2)
    candidates = {
        "logreg": LogisticRegression(max_iter=500, solver="liblinear"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    # Save final local artifact (for API loading)
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    mlflow.set_experiment("mlops-a1-heart-disease")

    best = {"name": None, "roc_auc": -1.0, "run_id": None}

    for model_name, model in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            clf = Pipeline(steps=[
                ("preprocess", preprocess.named_steps["preprocess"]),
                ("model", model),
            ])

            clf.fit(X_train, y_train)

            val_prob = clf.predict_proba(X_val)[:, 1]
            test_prob = clf.predict_proba(X_test)[:, 1]

            val_metrics = evaluate_binary(y_val, val_prob)
            test_metrics = evaluate_binary(y_test, test_prob)

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("numeric_cols", ",".join(numeric_cols))
            mlflow.log_param("categorical_cols", ",".join(categorical_cols))

            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    mlflow.log_param(f"model__{k}", str(v))

            for k, v in val_metrics.items():
                if k != "tn_fp_fn_tp":
                    mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                if k != "tn_fp_fn_tp":
                    mlflow.log_metric(f"test_{k}", v)

            cm_payload = {
                "val_confusion_tn_fp_fn_tp": val_metrics["tn_fp_fn_tp"],
                "test_confusion_tn_fp_fn_tp": test_metrics["tn_fp_fn_tp"],
            }

            cm_path = artifacts_dir / f"{model_name}_confusion.json"
            cm_path.write_text(json.dumps(cm_payload, indent=2))
            mlflow.log_artifact(str(cm_path))

            mlflow.sklearn.log_model(clf, artifact_path="model")

            if val_metrics["roc_auc"] > best["roc_auc"]:
                best = {
                    "name": model_name,
                    "roc_auc": val_metrics["roc_auc"],
                    "run_id": run.info.run_id,
                }

    best_path = artifacts_dir / "best_run.json"
    best_path.write_text(json.dumps(best, indent=2))
    print("Best model:", best)

    best_model_uri = f"runs:/{best['run_id']}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    joblib.dump(best_model, artifacts_dir / "model.joblib")
    print("Saved production model to artifacts/model.joblib")


if __name__ == "__main__":
    main()

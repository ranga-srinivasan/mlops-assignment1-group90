"""
FastAPI model-serving API.

- Loads a trained sklearn pipeline from artifacts/model.joblib
- Exposes /predict that accepts JSON and returns:
    - prediction (0/1)
    - probability (confidence)
- Includes request logging (basic MLOps requirement)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator


# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("heart_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# -------------------------
# Pydantic request schema
# -------------------------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, description="Age in years")
    sex: float = Field(
        ..., description="Sex (0=female, 1=male)"
    )
    cp: float = Field(
        ..., description="Chest pain type (1-4)"
    )
    trestbps: float = Field(
        ..., description="Resting blood pressure"
    )
    chol: float = Field(
        ..., description="Serum cholesterol (mg/dl)"
    )
    fbs: float = Field(
        ..., description="Fasting blood sugar > 120 mg/dl"
    )
    restecg: float = Field(
        ..., description="Resting ECG results (0-2)"
    )
    thalach: float = Field(
        ..., description="Maximum heart rate achieved"
    )
    exang: float = Field(
        ..., description="Exercise induced angina"
    )
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise"
    )
    slope: float = Field(
        ..., description="Slope of peak exercise ST segment"
    )
    ca: float = Field(
        ..., description="Number of major vessels (0-3)"
    )
    thal: float = Field(
        ..., description="Thalassemia (3, 6, 7)"
    )


class PredictResponse(BaseModel):
    prediction: int
    probability: float


# -------------------------
# App + model loading
# -------------------------
app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0"
)

MODEL = None


@app.on_event("startup")
def load_model() -> None:
    """
    Load the trained pipeline once when the container/app starts.
    """
    global MODEL
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "model.joblib"
    MODEL = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)


@app.get("/health")
def health() -> dict:
    """
    Health endpoint for Kubernetes readiness/liveness checks.
    """
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse
)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict endpoint:
    - Convert request to DataFrame
    - Run predict_proba
    - Return binary prediction + probability
    """
    payload = req.model_dump()
    df = pd.DataFrame([payload])

    logger.info("Received request: %s", payload)

    probability = float(MODEL.predict_proba(df)[:, 1][0])
    prediction = int(probability >= 0.5)

    logger.info(
        "Prediction=%s, Probability=%.4f",
        prediction,
        probability,
    )

    return PredictResponse(
        prediction=prediction,
        probability=probability,
    )


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        "%s %s | Status=%s | Duration=%.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


# -------------------------
# Prometheus instrumentation
# -------------------------
Instrumentator().instrument(app).expose(app)

"""
FastAPI model-serving API.

- Loads a trained sklearn pipeline from artifacts/model.joblib
- Exposes /predict that accepts JSON and returns:
    - prediction (0/1)
    - probability (confidence)
- Includes request logging (basic MLOps requirement)

This supports the assignment requirement for containerized API. :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import logging
from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import Request
import time

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("heart_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------------
# Pydantic request schema
# -------------------------
class PredictRequest(BaseModel):
    # Keep field names consistent with training columns
    age: float = Field(..., ge=0, description="Age in years")
    sex: float = Field(..., description="Sex (0=female, 1=male) typically")
    cp: float = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: float = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of the peak exercise ST segment (1-3)")
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: float = Field(..., description="Thalassemia (commonly 3,6,7)")


class PredictResponse(BaseModel):
    prediction: int
    probability: float


# -------------------------
# App + model loading
# -------------------------
app = FastAPI(title="Heart Disease Risk API", version="1.0")

MODEL = None

@app.on_event("startup")
def load_model():
    """
    Load the trained pipeline once when the container/app starts.
    """
    global MODEL
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "model.joblib"
    MODEL = joblib.load(model_path)
    logger.info(f"Loaded model from: {model_path}")


@app.get("/health")
def health():
    """
    Health endpoint for Kubernetes readiness/liveness checks.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict endpoint:
      - convert request to DataFrame
      - run predict_proba to get confidence
      - return binary prediction + probability
    """
    payload = req.model_dump()
    df = pd.DataFrame([payload])

    # Log request (be mindful: in real healthcare apps, avoid logging sensitive data)
    logger.info(f"Received request: {payload}")

    prob = float(MODEL.predict_proba(df)[:, 1][0])
    pred = int(prob >= 0.5)

    logger.info(f"Prediction={pred}, Probability={prob:.4f}")
    return PredictResponse(prediction=pred, probability=prob)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.3f}s"
    )

    return response


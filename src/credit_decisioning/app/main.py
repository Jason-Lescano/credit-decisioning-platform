"""
FastAPI app (Sprint 1)

Endpoints:
- GET /health
- POST /score

The /score endpoint loads the saved LightGBM artifact created by scripts/train_all.py
and returns:
- PD (probability of default)
- simple decision policy (approve/review/reject)
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from credit_decisioning.app.schemas import ScoreRequest, ScoreResponse

app = FastAPI(title="Credit Decisioning Platform")

MODEL_PATH = Path("artifacts/models/lgbm_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python scripts/train_all.py"
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_names"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        model, feature_names = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build one-row dataframe from incoming features
    X = pd.DataFrame([req.features])

    # One-hot encode similarly to training
    X = pd.get_dummies(X, dummy_na=True)

    # Align columns to training feature set (missing -> 0)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    proba = float(model.predict_proba(X)[0, 1])

    # Simple baseline policy (Sprint 2 becomes segment/policy-aware)
    if proba < 0.03:
        decision = "approve"
    elif proba < 0.08:
        decision = "review"
    else:
        decision = "reject"

    return ScoreResponse(pd=proba, decision=decision, reasons=["baseline_policy"])

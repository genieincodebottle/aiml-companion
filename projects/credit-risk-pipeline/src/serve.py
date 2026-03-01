"""
Model Serving (FastAPI)
=======================

FastAPI prediction endpoint for the credit risk model.
Loads the best trained model from ``artifacts/results/best_model.joblib``
and serves predictions via a REST API.

Usage::

    uvicorn src.serve:app --reload --port 8000

Endpoints
---------
GET  /health           Health check.
POST /predict          Single prediction.
POST /predict/batch    Batch predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "results" / "best_model.joblib"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit default probability using a trained ML pipeline.",
    version="0.1.0",
)

# Global model reference (loaded on startup)
_model = None


@app.on_event("startup")
def load_model() -> None:
    """Load the trained model on startup."""
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"No model found at {MODEL_PATH}. Train first.")


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class CreditApplication(BaseModel):
    """Single credit application for scoring."""
    duration: int = Field(..., ge=1, description="Loan duration in months")
    credit_amount: float = Field(..., gt=0, description="Loan amount")
    age: int = Field(..., ge=18, description="Applicant age")
    employment_since: Optional[float] = Field(None, description="Years at current job")
    income: Optional[float] = Field(None, description="Annual income")
    existing_credits: Optional[int] = Field(None, description="Number of existing credits")
    housing: Optional[str] = Field(None, description="Housing type (own/rent/free)")
    purpose: Optional[str] = Field(None, description="Loan purpose")

    class Config:
        json_schema_extra = {
            "example": {
                "duration": 24,
                "credit_amount": 5000,
                "age": 35,
                "employment_since": 4.0,
                "income": 45000,
                "existing_credits": 1,
                "housing": "own",
                "purpose": "car",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction result."""
    default_probability: float
    risk_category: str
    threshold: float
    prediction: int
    adverse_action_reasons: list[str]


class BatchRequest(BaseModel):
    """Batch prediction request."""
    applications: list[CreditApplication]


class BatchResponse(BaseModel):
    """Batch prediction result."""
    predictions: list[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """Score a single credit application."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    df = pd.DataFrame([application.model_dump()])
    return _score_single(df)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score a batch of credit applications."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    records = [app.model_dump() for app in request.applications]
    df = pd.DataFrame(records)

    predictions = []
    for i in range(len(df)):
        row_df = df.iloc[[i]]
        predictions.append(_score_single(row_df))

    return BatchResponse(predictions=predictions, count=len(predictions))


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.35


def _score_single(df: pd.DataFrame) -> PredictionResponse:
    """Score a single row DataFrame and return prediction."""
    try:
        proba = _model.predict_proba(df)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    prediction = int(proba >= DEFAULT_THRESHOLD)

    # Risk category
    if proba < 0.15:
        category = "Low Risk"
    elif proba < 0.35:
        category = "Medium Risk"
    elif proba < 0.60:
        category = "High Risk"
    else:
        category = "Very High Risk"

    # Adverse action reasons (top contributing features)
    reasons = _get_adverse_reasons(df, proba)

    return PredictionResponse(
        default_probability=round(float(proba), 4),
        risk_category=category,
        threshold=DEFAULT_THRESHOLD,
        prediction=prediction,
        adverse_action_reasons=reasons,
    )


def _get_adverse_reasons(df: pd.DataFrame, proba: float) -> list[str]:
    """Generate adverse action reasons based on feature values.

    In production, this would use SHAP values per-prediction.
    This simplified version uses rule-based reasons.
    """
    reasons = []
    row = df.iloc[0]

    if pd.notna(row.get("credit_amount")) and row["credit_amount"] > 10000:
        reasons.append("High loan amount requested")

    if pd.notna(row.get("duration")) and row["duration"] > 36:
        reasons.append("Long loan duration")

    if pd.notna(row.get("age")) and row["age"] < 25:
        reasons.append("Limited credit history (young applicant)")

    if pd.notna(row.get("existing_credits")) and row["existing_credits"] > 3:
        reasons.append("High number of existing credits")

    if pd.notna(row.get("income")) and pd.notna(row.get("credit_amount")):
        if row["credit_amount"] / max(row["income"], 1) > 0.3:
            reasons.append("High debt-to-income ratio")

    if not reasons and proba > DEFAULT_THRESHOLD:
        reasons.append("Combined risk factors exceed threshold")

    return reasons
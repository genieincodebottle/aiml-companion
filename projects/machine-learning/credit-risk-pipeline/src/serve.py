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

import json
import logging
from contextlib import asynccontextmanager
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
THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "results" / "threshold.json"

# Fallback when no tuned threshold artifact exists yet
DEFAULT_THRESHOLD = 0.35

# Globals loaded on startup
_model = None
_config: dict = {}
_expected_columns: list[str] = []
_categorical_columns: list[str] = []
_threshold: float = DEFAULT_THRESHOLD


def _load_artifacts() -> None:
    """Load model, config, schema, and tuned threshold."""
    global _model, _config, _expected_columns, _categorical_columns, _threshold

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")

        # Training schema: the ColumnTransformer requires these exact columns
        _expected_columns = list(_model.feature_names_in_)
        preprocessor = _model.named_steps["preprocessor"]
        for name, _, cols in preprocessor.transformers_:
            if name == "cat":
                _categorical_columns = list(cols)
    else:
        logger.warning(f"No model found at {MODEL_PATH}. Train first.")

    # Config drives serve-time feature engineering (same as training)
    try:
        from src.data_loader import load_config
        _config = load_config()
    except Exception as e:
        logger.warning(f"Could not load config, feature engineering disabled: {e}")
        _config = {}

    # Cost-optimal threshold from the evaluate stage, if available
    if THRESHOLD_PATH.exists():
        try:
            data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
            _threshold = float(data["threshold"])
            logger.info(f"Using tuned threshold {_threshold:.2f} from {THRESHOLD_PATH}")
        except Exception as e:
            logger.warning(f"Could not read threshold artifact, using default: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup."""
    _load_artifacts()
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit default probability using a trained ML pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

#: Features the model leans on hardest. Scoring without them is legal and
#: silent and produces a much weaker model than the evaluation report
#: describes, so their absence is reported rather than imputed away.
#:
#: Measured, cross-validated AUC:
#:     full training schema                    0.782
#:     only duration + credit_amount + age     0.636   <- what this API used
#:                                                        to accept
#:     checking_status ALONE                   0.680
#:
#: One categorical the API did not expose carried more signal than every
#: numeric field it did. The served model and the evaluated model were not the
#: same system.
KEY_FEATURES = ("checking_status", "credit_history", "savings_status",
                "employment", "purpose")


class CreditApplication(BaseModel):
    """Single credit application for scoring.

    The optional fields are optional only in the sense that the request will
    not be rejected without them. Omit the ones in KEY_FEATURES and the model
    is scoring mostly imputed values -- the response says so in
    `missing_key_features`.
    """
    duration: int = Field(..., ge=1, description="Loan duration in months")
    credit_amount: float = Field(..., gt=0, description="Loan amount")
    age: int = Field(..., ge=18, description="Applicant age")

    # --- the features that actually drive the score
    checking_status: Optional[str] = Field(
        None, description="Checking account status, e.g. '<0', '0<=X<200', "
                          "'>=200', 'no checking' -- the single strongest "
                          "predictor in this model")
    credit_history: Optional[str] = Field(
        None, description="e.g. 'critical/other existing credit', "
                          "'existing paid', 'delayed previously'")
    savings_status: Optional[str] = Field(
        None, description="e.g. '<100', '100<=X<500', 'no known savings'")
    employment: Optional[str] = Field(
        None, description="Years employed, e.g. '<1', '1<=X<4', '>=7'")
    purpose: Optional[str] = Field(None, description="Loan purpose")

    # --- minor extras
    employment_since: Optional[float] = Field(None, description="Years at current job")
    existing_credits: Optional[int] = Field(None, description="Number of existing credits")
    housing: Optional[str] = Field(None, description="Housing type (own/rent/free)")
    income: Optional[float] = Field(
        None, description="Annual income. NOTE: the German Credit training "
                          "data has no income column, so this is accepted for "
                          "forward compatibility and does NOT affect the "
                          "score.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "duration": 24,
                "credit_amount": 5000,
                "age": 35,
                "checking_status": "<0",
                "credit_history": "existing paid",
                "savings_status": "<100",
                "employment": "1<=X<4",
                "purpose": "car",
                "housing": "own",
                "existing_credits": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result."""
    default_probability: float
    risk_category: str
    threshold: float
    prediction: int
    adverse_action_reasons: list[str]
    #: Key drivers absent from the request and therefore imputed. A non-empty
    #: list means this score is weaker than the evaluation report implies.
    missing_key_features: list[str] = []


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
        "threshold": _threshold,
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

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Align a request DataFrame with the model's training schema.

    The pipeline was trained on the full dataset schema plus engineered
    features, so we (1) run the same feature engineering, (2) reindex to
    the training columns (absent ones become NaN and are handled by the
    KNN imputer), and (3) fill absent categoricals with a placeholder
    that the OneHotEncoder zero-encodes via handle_unknown='ignore'.
    """
    if _config:
        try:
            from src.features import engineer_features
            df = engineer_features(df, _config)
        except Exception as e:
            logger.warning(f"Serve-time feature engineering failed: {e}")

    df = df.reindex(columns=_expected_columns)
    if _categorical_columns:
        df[_categorical_columns] = (
            df[_categorical_columns].astype(object).where(
                df[_categorical_columns].notna(), "missing"
            ).astype(str)
        )
    return df


def _score_single(df: pd.DataFrame) -> PredictionResponse:
    """Score a single row DataFrame and return prediction."""
    # Recorded BEFORE _prepare_features imputes them out of existence.
    missing_key = [c for c in KEY_FEATURES
                   if c not in df.columns or df[c].isna().all()]
    try:
        features = _prepare_features(df)
        proba = _model.predict_proba(features)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    prediction = int(proba >= _threshold)

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
        threshold=_threshold,
        prediction=prediction,
        adverse_action_reasons=reasons,
        missing_key_features=missing_key,
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

    if not reasons and proba > _threshold:
        reasons.append("Combined risk factors exceed threshold")

    return reasons
"""
Production Model Server with FastAPI.

Lifespan model loading, graceful shutdown, Pydantic validation, health checks.

Usage:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.app.metrics import track_error, track_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to project root so the server starts
# regardless of the working directory uvicorn was launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config():
    """Load configs/base.yaml; fall back to defaults if absent."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config not found at {CONFIG_PATH}, using defaults")
    return {}


_config = load_config()
MODEL_PATH = PROJECT_ROOT / _config.get("model", {}).get("path", "artifacts/models/model.joblib")

model = None
# Environment beats config so docker-compose can pin a version per deploy
model_version = os.environ.get("MODEL_VERSION") or _config.get("model", {}).get("version", "v1.0.0")
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global model, start_time
    logger.info("Loading model at startup...")
    try:
        model = joblib.load(MODEL_PATH)
        start_time = time.time()
        logger.info(f"Model {model_version} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    yield
    logger.info("Shutting down gracefully...")


app = FastAPI(title="ML Model Server", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    probability: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check - used by Docker HEALTHCHECK and load balancers."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        uptime_seconds=time.time() - start_time if start_time else 0
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Serve predictions with input validation."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        t0 = time.time()
        features = np.array(request.features).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features).max())
        latency = time.time() - t0
        track_prediction(model_version, prediction, latency)
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=model_version
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        track_error(type(e).__name__)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def shutdown_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

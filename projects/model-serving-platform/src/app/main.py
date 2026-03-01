"""
Production Model Server with FastAPI.

Lifespan model loading, graceful shutdown, Pydantic validation, health checks.

Usage:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000
"""
import signal
import sys
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager
import joblib
import logging
from src.app.metrics import track_prediction, track_error, PREDICTION_LATENCY, PREDICTION_COUNT, ERROR_COUNT
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
model_version = "v1.0.0"
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global model, start_time
    logger.info("Loading model at startup...")
    try:
        model = joblib.load("artifacts/models/model.joblib")
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

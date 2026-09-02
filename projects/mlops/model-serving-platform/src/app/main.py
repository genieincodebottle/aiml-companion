"""
Production Model Server with FastAPI.

Lifespan model loading, graceful shutdown, Pydantic validation, health checks.

Usage:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
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
async def health(response: Response):
    """Readiness check - used by Docker HEALTHCHECK and load balancers.

    Returns **503** when the model is not loaded, not 200 with a sad string in
    the body. This matters more than it looks: `curl -f` (what the Dockerfile
    and docker-compose healthchecks use) fails only on HTTP >= 400, and load
    balancers route on status code. This endpoint used to answer 200 with
    `status="unhealthy"`, so a server with no model was reported healthy and
    kept receiving production traffic -- every request then failing with a 503
    from /predict. The one component whose job is to notice the outage was the
    one reporting everything fine.

    This is a READINESS probe: "can I serve requests right now". Liveness --
    "is the process wedged, restart me" -- is a different question, and in
    Kubernetes it belongs on a separate endpoint that does not depend on the
    model, or a crash-looping pod that fails to load a model will be restarted
    forever instead of being marked unready and left alone.
    """
    ready = model is not None
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        model_loaded=ready,
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


# NOTE: no signal handlers are installed here, deliberately.
#
# This module used to register SIGTERM/SIGINT handlers that called sys.exit(0)
# at import time. That defeats the graceful shutdown the lifespan block exists
# to provide: uvicorn installs its own handlers, which stop accepting new
# connections, let in-flight requests finish, and then run the lifespan
# shutdown. Overriding them with an immediate sys.exit kills the process
# mid-request and skips the "Shutting down gracefully" path entirely -- so the
# code advertising graceful shutdown was the code preventing it.
#
# Registering handlers at import time is also wrong on its own terms: signal
# handlers can only be set from the main thread, so importing this module from
# a worker thread or a test runner raises ValueError.
#
# If you need cleanup on shutdown, put it after the `yield` in lifespan().

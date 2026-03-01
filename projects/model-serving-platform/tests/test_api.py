"""Tests for MLOps Model Server. Run: pytest tests/ -v"""
import pytest


def test_app_imports():
    """Test that app module imports correctly."""
    from src.app.main import app, PredictionRequest, HealthResponse
    assert app is not None


def test_prediction_request_validation():
    """Test Pydantic model validation."""
    from src.app.main import PredictionRequest
    req = PredictionRequest(features=[1.0, 2.0, 3.0, 4.0])
    assert len(req.features) == 4


def test_prediction_request_empty_rejected():
    """Test that empty features are rejected."""
    from src.app.main import PredictionRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictionRequest(features=[])


def test_metrics_imports():
    """Test metrics module imports."""
    from src.app.metrics import PREDICTION_LATENCY, PREDICTION_COUNT, track_prediction
    assert PREDICTION_LATENCY is not None
    assert PREDICTION_COUNT is not None


def test_health_response_model():
    """Test HealthResponse model."""
    from src.app.main import HealthResponse
    resp = HealthResponse(
        status="healthy", model_loaded=True,
        model_version="v1.0.0", uptime_seconds=100.0
    )
    assert resp.status == "healthy"

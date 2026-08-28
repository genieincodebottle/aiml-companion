"""Tests for MLOps Model Server. Run: pytest tests/ -v"""
import pytest


def test_app_imports():
    """Test that app module imports correctly."""
    from src.app.main import app
    assert app is not None


def test_prediction_request_validation():
    """Test Pydantic model validation."""
    from src.app.main import PredictionRequest
    req = PredictionRequest(features=[1.0, 2.0, 3.0, 4.0])
    assert len(req.features) == 4


def test_prediction_request_empty_rejected():
    """Test that empty features are rejected."""
    from pydantic import ValidationError

    from src.app.main import PredictionRequest
    with pytest.raises(ValidationError):
        PredictionRequest(features=[])


def test_metrics_imports():
    """Test metrics module imports."""
    from src.app.metrics import PREDICTION_COUNT, PREDICTION_LATENCY
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


# ---------------------------------------------------------------------------
# Live endpoint tests (TestClient with lifespan)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with the model loaded via lifespan.

    Creates the demo model first if it does not exist yet, so the
    suite passes on a fresh clone / in CI without a manual setup step.
    """
    from fastapi.testclient import TestClient

    from src.app.main import MODEL_PATH, app

    if not MODEL_PATH.exists():
        import joblib
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        X, y = load_iris(return_X_y=True)
        demo = RandomForestClassifier(n_estimators=10, random_state=42)
        demo.fit(X, y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(demo, MODEL_PATH)

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Health endpoint reports healthy with model loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["uptime_seconds"] >= 0


def test_predict_endpoint(client):
    """Predict returns a class and probability for valid features."""
    resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1, 2)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "v1.0.0"


def test_predict_wrong_feature_count(client):
    """Wrong feature count is a 400 (model rejects), not a server crash."""
    resp = client.post("/predict", json={"features": [1.0, 2.0]})
    assert resp.status_code == 400


def test_predict_empty_features_rejected(client):
    """Empty features fail Pydantic validation with 422."""
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    """Metrics endpoint exposes Prometheus counters after a prediction."""
    client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "predictions_total" in resp.text
    assert "prediction_latency_seconds" in resp.text

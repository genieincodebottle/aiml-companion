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


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

def test_psi_detects_drift_that_leaves_the_reference_range():
    """The monitor must be loudest when drift is worst, not silent.

    Regression test for a defect that made the drift monitor actively
    dangerous. np.histogram drops out-of-range values, and because PSI bins are
    reference QUANTILES the reference histogram is uniform by construction --
    so a production distribution sharing no support with reference produced an
    all-zero (hence uniform) histogram and PSI ~ 0.0 "stable".

    Measured on the old code with reference ~ N(0,1):
        N(0.5, 1)  -> 0.2492 moderate_shift
        N(2, 1)    -> 3.1664 significant_shift
        N(10, 1)   -> 0.0000 STABLE          <- catastrophic drift, no alarm
        N(100, 1)  -> 0.0000 STABLE
        constant   -> 0.0000 STABLE
    """
    import numpy as np
    from src.monitoring.metrics import compute_psi, classify_psi

    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)

    same = compute_psi(reference, rng.normal(0, 1, 5000))
    assert classify_psi(same) == "stable", "identical distributions must not alarm"

    # The cases that used to report 0.0
    for shift in (10, 100):
        psi = compute_psi(reference, rng.normal(shift, 1, 5000))
        assert psi > 1.0, f"shift of {shift} reported PSI={psi}"
        assert classify_psi(psi) == "significant_shift"

    constant = compute_psi(reference, np.full(5000, 999.0))
    assert classify_psi(constant) == "significant_shift"

    # PSI must not fall as drift gets worse
    mild = compute_psi(reference, rng.normal(0.5, 1, 5000))
    severe = compute_psi(reference, rng.normal(3, 1, 5000))
    extreme = compute_psi(reference, rng.normal(50, 1, 5000))
    assert mild < severe < extreme, (
        f"PSI is not monotone in drift severity: {mild:.3f}, {severe:.3f}, "
        f"{extreme:.3f}")


def test_psi_is_zero_for_the_same_data():
    """Sanity anchor: a distribution cannot have drifted from itself."""
    import numpy as np
    from src.monitoring.metrics import compute_psi

    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, 2000)
    assert compute_psi(data, data) < 1e-6

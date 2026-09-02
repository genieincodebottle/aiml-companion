"""Tests for the FastAPI serving layer.

Endpoint tests need a trained model artifact, so they are skipped
until ``python main.py`` has been run at least once.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serve import MODEL_PATH, app

MODEL_TRAINED = MODEL_PATH.exists()

needs_model = pytest.mark.skipif(
    not MODEL_TRAINED,
    reason="No trained model artifact. Run `python main.py` first.",
)


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan startup (loads model + threshold)."""
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] == MODEL_TRAINED


@needs_model
def test_predict_minimal_fields(client):
    """The README's 4-field example must score successfully."""
    resp = client.post(
        "/predict",
        json={"duration": 24, "credit_amount": 5000, "age": 35, "income": 45000},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_category"] in (
        "Low Risk", "Medium Risk", "High Risk", "Very High Risk"
    )


@needs_model
def test_predict_full_fields(client):
    resp = client.post(
        "/predict",
        json={
            "duration": 24,
            "credit_amount": 5000,
            "age": 35,
            "employment_since": 4.0,
            "income": 45000,
            "existing_credits": 1,
            "housing": "own",
            "purpose": "car",
        },
    )
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["default_probability"] <= 1.0


@needs_model
def test_predict_batch(client):
    resp = client.post(
        "/predict/batch",
        json={
            "applications": [
                {"duration": 12, "credit_amount": 2000, "age": 45, "income": 60000},
                {"duration": 48, "credit_amount": 15000, "age": 22, "income": 20000},
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


@needs_model
def test_risky_applicant_scores_higher(client):
    """A large long loan for a young low-income applicant should score
    higher than a small short loan for an established applicant."""
    safe = client.post(
        "/predict",
        json={"duration": 12, "credit_amount": 2000, "age": 45, "income": 60000},
    ).json()
    risky = client.post(
        "/predict",
        json={"duration": 48, "credit_amount": 15000, "age": 22, "income": 20000},
    ).json()
    assert risky["default_probability"] > safe["default_probability"]


def test_predict_rejects_invalid_age(client):
    resp = client.post(
        "/predict",
        json={"duration": 24, "credit_amount": 5000, "age": 15, "income": 45000},
    )
    assert resp.status_code == 422


def test_predict_rejects_missing_required(client):
    resp = client.post("/predict", json={"age": 35})
    assert resp.status_code == 422


@needs_model
def test_missing_key_features_are_reported_not_hidden(client):
    """Scoring without the model's main drivers must be visible to the caller.

    The API used to accept duration, credit_amount, age and income. Income is
    not even in the training data, and those three fields carry a
    cross-validated AUC of 0.636 against 0.782 for the full schema --
    `checking_status` ALONE scores 0.680. Requests were silently imputed up to
    the training schema, so the served model was much weaker than the
    evaluation report described, and nothing said so.
    """
    sparse = client.post(
        "/predict",
        json={"duration": 48, "credit_amount": 15000, "age": 22},
    ).json()
    assert sparse["missing_key_features"], (
        "a request lacking every key driver must report them")

    rich = client.post(
        "/predict",
        json={"duration": 48, "credit_amount": 15000, "age": 22,
              "checking_status": "<0", "credit_history": "delayed previously",
              "savings_status": "<100", "employment": "<1",
              "purpose": "new car"},
    ).json()
    assert rich["missing_key_features"] == []
    # the key features must actually move the score
    assert rich["default_probability"] != sparse["default_probability"]

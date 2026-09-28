"""FastAPI surface: health, catalog, run creation, and error paths."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_reports_stub_mode():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "stub"          # no key / SDK in tests
    assert body["sdk_available"] is False


def test_jobs_catalog():
    jobs = client.get("/api/jobs").json()["jobs"]
    assert any(j["id"] == "datetime-fleet" and j["repo_count"] == 4 for j in jobs)


def test_create_run_returns_id():
    r = client.post("/api/runs", json={"job_id": "datetime-fleet", "auto_approve": True})
    assert r.status_code == 200
    assert r.json()["run_id"].startswith("job_")


def test_unknown_job_is_400():
    assert client.post("/api/runs", json={"job_id": "does-not-exist"}).status_code == 400


def test_unknown_run_is_404():
    assert client.get("/api/runs/nope").status_code == 404


def test_approval_without_pending_is_409():
    rid = client.post("/api/runs", json={"job_id": "datetime-fleet", "auto_approve": True}).json()["run_id"]
    r = client.post(f"/api/runs/{rid}/approve", json={"repo_id": "billing-service", "decision": "approve"})
    assert r.status_code == 409

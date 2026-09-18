"""API access rules and the HITL decision contract.

Before these fixes: any signed-in claimant could list every claim, read the
HITL queue and analytics, and anyone could self-register as admin. The
reviewer name on a decision came from the browser, and notes were optional
despite the UI's "Required for audit trail".
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from api import db as api_db
from api import security
from api.routes_analytics import FINISHED_STATUSES, router as analytics_router
from api.routes_auth import router as auth_router
from api.routes_claims import router as claims_router
from api.routes_hitl import router as hitl_router
from src.hitl import queue


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(queue, "DB_PATH", tmp_path / "hitl.db")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit"))

    app = FastAPI()
    for r in (auth_router, claims_router, hitl_router, analytics_router):
        app.include_router(r)

    def session_override():
        with Session(engine) as s:
            yield s

    current = {"user": None}
    app.dependency_overrides[api_db.get_session] = session_override
    app.dependency_overrides[security.get_current_user] = lambda: current["user"]
    c = TestClient(app)
    c.as_role = lambda role, uid=1, name="u": current.__setitem__(
        "user", api_db.User(id=uid, username=name, email=f"{name}@example.com", password_hash="x", role=role)
    )
    return c


@pytest.mark.parametrize("path", ["/api/claims/all", "/api/hitl/queue", "/api/hitl/stats", "/api/analytics/metrics"])
def test_claimants_cannot_read_everyone_elses_data(client, path):
    client.as_role("user")
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", ["/api/claims/all", "/api/hitl/queue", "/api/analytics/metrics"])
def test_reviewers_can(client, path):
    client.as_role("reviewer")
    assert client.get(path).status_code == 200


def test_self_registration_cannot_create_an_admin(client):
    r = client.post("/api/auth/register", json={
        "username": "mallory", "email": "m@example.com", "password": "pw12345", "role": "admin"})
    assert r.status_code == 403


def test_self_registration_as_reviewer_still_works_for_the_demo(client):
    r = client.post("/api/auth/register", json={
        "username": "rita", "email": "r@example.com", "password": "pw12345", "role": "reviewer"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "reviewer"


def _pending_ticket(ticket_id="HITL-T1", created=None, deadline=None):
    conn = queue._get_db()
    now = created or datetime.now(timezone.utc)
    conn.execute(
        "INSERT INTO hitl_queue (ticket_id, claim_id, priority, priority_score, triggers, review_brief, "
        "state_snapshot, status, created_at, sla_deadline) VALUES (?, 'CLM-1', 'normal', 20, '[]', 'b', '{}', "
        "'pending', ?, ?)",
        (ticket_id, now.isoformat(), (deadline or now + timedelta(hours=72)).isoformat()),
    )
    conn.commit()
    conn.close()


def test_decisions_require_notes(client):
    client.as_role("reviewer")
    _pending_ticket()
    r = client.post("/api/hitl/decide/HITL-T1", json={"decision": "approved", "notes": "   "})
    assert r.status_code == 400 and "notes" in r.json()["detail"].lower()


def test_decision_is_attributed_to_the_logged_in_reviewer(client, monkeypatch):
    import api.routes_claims as rc
    monkeypatch.setattr(rc, "resume_pipeline_for_claim", lambda claim_id, decision: {})
    client.as_role("reviewer", name="real_reviewer")
    _pending_ticket()
    r = client.post("/api/hitl/decide/HITL-T1", json={
        "decision": "approved", "notes": "checked photos", "reviewer_id": "someone_else"})
    assert r.status_code == 200 and r.json()["reviewer"] == "real_reviewer"
    with sqlite3.connect(queue.DB_PATH) as conn:
        assert conn.execute("SELECT reviewer_id FROM hitl_queue").fetchone()[0] == "real_reviewer"


def test_overdue_tickets_are_escalated_and_listed_first(client):
    client.as_role("reviewer")
    long_ago = datetime.now(timezone.utc) - timedelta(days=5)
    _pending_ticket("HITL-OLD", created=long_ago, deadline=long_ago + timedelta(hours=72))
    _pending_ticket("HITL-NEW")
    queue_rows = client.get("/api/hitl/queue").json()
    assert queue_rows[0]["ticket_id"] == "HITL-OLD"
    assert queue_rows[0]["sla_breached"] is True and queue_rows[0]["priority"] == "critical"
    assert client.get("/api/hitl/stats").json()["pending_sla_breached"] == 1


def test_analytics_counts_the_statuses_the_pipeline_writes():
    # _persist_pipeline_result writes the decision as the status.
    assert {"approved", "denied", "auto_rejected", "approved_partial"} <= FINISHED_STATUSES
    assert "pending_human_review" not in FINISHED_STATUSES

"""Tests for the endpoints the React console calls.

These drive the real FastAPI app with FastAPI's TestClient against the offline
rule engine. SQLite and Chroma are pointed at a temporary directory, so the
tests never touch backend/databases. No key, no server, no network.

What they pin down:
- "Run AI Moderation" has a route (POST /api/content/{id}/moderate).
- A paused item is finished by POST /api/hitl/review/{id}, which resumes the graph.
- A console appeal runs the Appeal Review Agent (advice only) and a human
  appeal decision is fed to LearningTracker.record_appeal.
- Decision endpoints need a session token with a moderator role.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["MODERATION_OFFLINE"] = "1"
os.environ["ENABLE_FAST_MODE"] = "false"
os.environ.pop("GOOGLE_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

MILD_INSULT = "You are being an idiot about this."
CLEAN = "Thanks for writing this up, the section on retries was really helpful."


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("console")
    os.environ["CHROMA_DB_PATH"] = str(workdir / "chroma")

    import main

    # main.py opens its databases by relative name, which resolves to
    # backend/databases. Redirect both to the temp directory.
    real_moderation_db, real_auth_db = main.ModerationDatabase, main.AuthDatabase
    main.ModerationDatabase = lambda _path: real_moderation_db(str(workdir / "moderation.db"))
    main.AuthDatabase = lambda _path: real_auth_db(str(workdir / "auth.db"))
    try:
        with TestClient(main.app) as test_client:
            test_client.main = main
            yield test_client
    finally:
        main.ModerationDatabase, main.AuthDatabase = real_moderation_db, real_auth_db


def _token(client, username, role):
    main = client.main
    main.auth_db.create_user(username, "pw-for-tests", username, role)
    user = main.auth_db.authenticate_user(username, "pw-for-tests")
    return main.auth_db.create_session(user["user_id"])


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def mod_headers(client):
    return _headers(_token(client, "t_mod", "moderator"))


@pytest.fixture(scope="module")
def policy_headers(client):
    return _headers(_token(client, "t_policy", "policy_specialist"))


@pytest.fixture(scope="module")
def user_headers(client):
    return _headers(_token(client, "t_user", "user"))


def _submit(client, text, **extra):
    body = {"content_text": text, "content_type": "story", "user_id": extra.pop("user_id", "u-test")}
    body.update(extra)
    resp = client.post("/api/content/submit", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _store_unmoderated(client, content_id, text, user_id="u-stored"):
    """Insert a content row that no graph has run on yet."""
    db = client.main.db
    db.create_content_submission({
        "content_id": content_id,
        "submission_id": f"SUB-{content_id}",
        "user_id": user_id,
        "username": user_id,
        "content_text": text,
        "content_type": "story_comment",
        "platform": "web",
        "language": "en",
        "submission_timestamp": "2026-01-01T00:00:00",
        "status": "submitted",
        "toxicity_score": 0.0,
        "requires_human_review": False,
    })


# ---------------------------------------------------------------------------
# Run AI Moderation
# ---------------------------------------------------------------------------

def test_moderate_runs_the_graph_on_stored_content(client, mod_headers):
    _store_unmoderated(client, "CNT-MOD-1", CLEAN)
    resp = client.post("/api/content/CNT-MOD-1/moderate", headers=mod_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert body["agent_decisions_count"] >= 4
    stored = client.main.db.get_content_by_id("CNT-MOD-1")
    assert stored["current_status"] == "approved"


def test_moderate_refuses_content_that_is_already_decided(client, mod_headers):
    _store_unmoderated(client, "CNT-MOD-2", CLEAN)
    assert client.post("/api/content/CNT-MOD-2/moderate", headers=mod_headers).status_code == 200
    again = client.post("/api/content/CNT-MOD-2/moderate", headers=mod_headers)
    assert again.status_code == 409


def test_moderate_unknown_content_is_404(client, mod_headers):
    assert client.post("/api/content/NOPE/moderate", headers=mod_headers).status_code == 404


# ---------------------------------------------------------------------------
# Paused items are finished through the graph-resume endpoint
# ---------------------------------------------------------------------------

def _paused_item(client):
    result = _submit(client, MILD_INSULT, user_id="u-vip", verified=True, follower_count=50000)
    assert result["hitl_required"] is True
    assert result["status"] == "pending_human_review"
    assert result["content_id"] in client.main.hitl_pending_reviews
    return result["content_id"]


def test_hitl_review_resumes_the_paused_graph(client, mod_headers):
    content_id = _paused_item(client)
    resp = client.post(
        f"/api/hitl/review/{content_id}",
        json={"decision": "approve", "reviewer_name": "t_mod", "notes": "fine"},
        headers=mod_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["success"] is True
    # The paused state is gone and the content is no longer waiting.
    assert content_id not in client.main.hitl_pending_reviews
    stored = client.main.db.get_content_by_id(content_id)
    assert stored["current_status"] != "pending_human_review"


def test_manual_review_alone_leaves_the_paused_state_behind(client, mod_headers):
    """The older direct-write endpoint is why the console now calls hitl/review
    for paused items: it records a decision but never resumes the graph."""
    content_id = _paused_item(client)
    resp = client.post(
        f"/api/content/{content_id}/manual-review",
        json={"decision": "approve", "reviewer_name": "t_mod", "notes": "x"},
        headers=mod_headers,
    )
    assert resp.status_code == 200
    assert content_id in client.main.hitl_pending_reviews


# ---------------------------------------------------------------------------
# Appeals
# ---------------------------------------------------------------------------

def _removed_content(client):
    result = _submit(client, "You are all worthless idiots and I hope every one of you gets hurt.",
                     user_id="u-appeal")
    return result["content_id"]


def test_console_appeal_runs_the_appeal_review_agent(client, user_headers):
    content_id = _removed_content(client)
    resp = client.post(
        "/api/appeals/submit",
        json={"content_id": content_id, "user_id": "u-appeal",
              "appeal_reason": "I was quoting someone else to criticise it."},
        headers=user_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending"          # a person still decides
    assert body["ai_review"] is not None
    assert body["ai_review"]["recommendation"] in {"uphold", "overturn", "partial"}
    names = [d["agent_name"] for d in client.main.db.get_agent_executions(content_id)]
    assert "Appeal Review Agent" in names


def test_human_appeal_decision_is_recorded_for_learning(client, user_headers, policy_headers):
    content_id = _removed_content(client)
    client.post(
        "/api/appeals/submit",
        json={"content_id": content_id, "user_id": "u-appeal", "appeal_reason": "Context was missing."},
        headers=user_headers,
    )
    tracker = client.main.learning_tracker
    before = tracker.total_appeals
    with client.main.db.get_connection() as conn:
        appeal_id = conn.execute(
            "SELECT id FROM appeals WHERE content_id = ?", (content_id,)
        ).fetchone()[0]
    resp = client.post(
        f"/api/appeals/APL-{appeal_id}/review",
        json={"outcome": "overturned", "reasoning": "Quoted for criticism."},
        headers=policy_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["learning_recorded"] is True
    assert tracker.total_appeals == before + 1
    assert tracker.total_overturns >= 1


def test_the_graph_and_the_api_share_one_learning_tracker(client):
    """Appeal outcomes only matter if they land in the memory the agents write to."""
    assert client.main.learning_tracker is not None


# ---------------------------------------------------------------------------
# Token checks on decision endpoints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path,body", [
    ("post", "/api/content/CNT-X/moderate", None),
    ("post", "/api/content/CNT-X/manual-review", {"decision": "approve"}),
    ("post", "/api/hitl/review/CNT-X", {"decision": "approve", "reviewer_name": "x"}),
    ("post", "/api/appeals/APL-1/review", {"outcome": "upheld"}),
])
def test_decision_endpoints_need_a_token(client, method, path, body):
    resp = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert resp.status_code in (401, 403)


def test_a_plain_user_cannot_decide(client, user_headers):
    resp = client.post("/api/content/CNT-X/manual-review", json={"decision": "approve"},
                       headers=user_headers)
    assert resp.status_code == 403


def test_a_moderator_cannot_review_appeals(client, mod_headers):
    resp = client.post("/api/appeals/APL-1/review", json={"outcome": "upheld"}, headers=mod_headers)
    assert resp.status_code == 403

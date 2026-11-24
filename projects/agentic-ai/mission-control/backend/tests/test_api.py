"""API surface: health, catalog, the background run, operators, maker-checker,
decision status codes and the live-without-a-key refusal."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import main
from src.config import reset_settings_cache
from src.control_plane import reset_control_plane

ANA = {"X-Operator": "ana"}
REX = {"X-Operator": "rex"}


def _client() -> TestClient:
    return TestClient(main.app)


def _wait(c: TestClient, wid: str, statuses=("awaiting_approval",), timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = c.get(f"/api/investigations/{wid}").json()
        if view["status"] in statuses:
            return view
        time.sleep(0.05)
    raise AssertionError(f"{wid} never reached {statuses}: {view}")


def _start(c: TestClient, alert="ALRT-001", headers=ANA) -> str:
    res = c.post("/api/investigations", json={"alert_id": alert}, headers=headers)
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "running"
    return res.json()["workflow_id"]


def _relogin(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reset_settings_cache()
    reset_control_plane()


def test_health_reports_offline_mode():
    body = _client().get("/api/health").json()
    assert body["status"] == "ok" and body["mode"] == "offline" and body["can_run"] is True
    assert body["policy_enabled"] is True and body["login_required"] is False


def test_catalog_endpoints():
    c = _client()
    assert len(c.get("/api/alerts").json()["alerts"]) == 5
    assert len(c.get("/api/agents").json()["agents"]) == 6
    names = {p["name"] for p in c.get("/api/prompts").json()["prompts"]}
    assert "fincrime.narrative" in names


def test_start_returns_at_once_and_the_case_runs_in_the_background():
    c = _client()
    wid = _start(c)
    view = _wait(c, wid)
    assert view["pending_action"]["tool"] == "case.file_sar"
    assert view["started_by"] == "ana"
    types = [e["type"] for e in c.get(f"/api/investigations/{wid}/events").json()["events"]]
    assert types[0] == "workflow_started" and "approval_requested" in types


def test_a_different_reviewer_approves_and_is_recorded():
    c = _client()
    wid = _start(c)
    _wait(c, wid)
    res = c.post("/api/decisions", json={"workflow_id": wid, "decision": "approve", "note": "ok"}, headers=REX)
    assert res.status_code == 200 and res.json()["status"] == "resolved"
    view = c.get(f"/api/investigations/{wid}").json()
    assert (view["disposition"], view["decided_by"], view["decision_note"]) == ("file_sar", "rex", "ok")


def test_maker_checker_is_a_403_over_http():
    c = _client()
    wid = _start(c)
    _wait(c, wid)
    res = c.post("/api/decisions", json={"workflow_id": wid, "decision": "approve"}, headers=ANA)
    assert res.status_code == 403 and "maker-checker" in res.json()["detail"]


def test_double_click_is_idempotent_and_a_conflicting_decision_is_409():
    c = _client()
    wid = _start(c)
    _wait(c, wid)
    body = {"workflow_id": wid, "decision": "approve"}
    first, second = c.post("/api/decisions", json=body, headers=REX), c.post("/api/decisions", json=body, headers=REX)
    assert first.json()["duplicate"] is False and second.json()["duplicate"] is True
    conflict = c.post("/api/decisions", json={"workflow_id": wid, "decision": "reject"}, headers=REX)
    assert conflict.status_code == 409 and "already decided" in conflict.json()["detail"]


@pytest.mark.parametrize(
    "body,code",
    [
        ({"workflow_id": "wf-nope", "decision": "approve"}, 404),
        ({"workflow_id": "wf-nope", "decision": "maybe"}, 400),
        ({"workflow_id": "../../etc", "decision": "approve"}, 404),
    ],
)
def test_bad_decisions_get_precise_errors(body, code):
    assert _client().post("/api/decisions", json=body, headers=REX).status_code == code


def test_unknown_alert_404_and_unknown_workflow_404():
    c = _client()
    assert c.post("/api/investigations", json={"alert_id": "NOPE"}, headers=ANA).status_code == 404
    assert c.get("/api/investigations/wf-nope").status_code == 404
    assert c.get("/api/investigations/wf-nope/events").status_code == 404
    assert c.get("/api/investigations/wf-nope/stream").status_code == 404


def test_a_bad_operator_name_is_rejected():
    res = _client().post("/api/investigations", json={"alert_id": "ALRT-001"}, headers={"X-Operator": "a b<script>"})
    assert res.status_code == 400


def test_the_case_list_includes_new_cases():
    c = _client()
    wid = _start(c, "ALRT-003")
    _wait(c, wid)
    assert wid in [v["workflow_id"] for v in c.get("/api/investigations").json()["investigations"]]


def test_the_stream_ends_with_a_done_frame_at_the_gate():
    c = _client()
    wid = _start(c)
    _wait(c, wid)
    text = c.get(f"/api/investigations/{wid}/stream").text
    assert "event: approval_requested" in text and "event: done" in text


def test_live_mode_without_a_key_refuses_to_start(monkeypatch):
    _relogin(monkeypatch, MC_MODE="live")
    c = _client()
    health = c.get("/api/health").json()
    assert health["can_run"] is False and health["status"] == "degraded"
    res = c.post("/api/investigations", json={"alert_id": "ALRT-001"}, headers=ANA)
    assert res.status_code == 503 and "GOOGLE_API_KEY" in res.json()["detail"]


def test_with_operators_configured_tokens_and_roles_are_enforced(monkeypatch):
    _relogin(monkeypatch, MC_OPERATORS="ana:analyst:tok-ana,rex:reviewer:tok-rex,boss:analyst+reviewer:tok-boss")
    c = _client()
    assert c.get("/api/health").json()["login_required"] is True
    # No token, a wrong token, and a header that only claims a name: all refused.
    assert c.post("/api/investigations", json={"alert_id": "ALRT-001"}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert c.post("/api/investigations", json={"alert_id": "ALRT-001"}, headers=bad).status_code == 401
    assert c.post("/api/investigations", json={"alert_id": "ALRT-001"}, headers=ANA).status_code == 401

    def auth(t):
        return {"Authorization": f"Bearer {t}"}

    # A reviewer may not open a case; an analyst may not decide one.
    assert c.post("/api/investigations", json={"alert_id": "ALRT-001"}, headers=auth("tok-rex")).status_code == 403
    wid = _start(c, headers=auth("tok-ana"))
    _wait(c, wid)
    as_analyst = c.post("/api/decisions", json={"workflow_id": wid, "decision": "approve"}, headers=auth("tok-ana"))
    assert as_analyst.status_code == 403
    assert c.get("/api/me", headers=auth("tok-rex")).json()["operator"] == {
        "name": "rex",
        "roles": ["reviewer"],
        "verified": True,
    }
    # Holding both roles does not get round maker-checker.
    wid2 = _start(c, headers=auth("tok-boss"))
    _wait(c, wid2)
    own = c.post("/api/decisions", json={"workflow_id": wid2, "decision": "approve"}, headers=auth("tok-boss"))
    assert own.status_code == 403
    ok = c.post("/api/decisions", json={"workflow_id": wid2, "decision": "approve"}, headers=auth("tok-rex"))
    assert ok.status_code == 200 and ok.json()["view"]["decided_by"] == "rex"

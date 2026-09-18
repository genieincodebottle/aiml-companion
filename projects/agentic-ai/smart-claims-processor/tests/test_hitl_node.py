"""The HITL node inside a real LangGraph graph: it pauses, it resumes, and the
resume does not create a second review ticket.

LangGraph re-runs a node from its first line on resume, so every side effect
before interrupt() runs twice per pause. Before the idempotency key, each
resume inserted a duplicate ticket into the review queue.
"""

import sqlite3

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents.graph import hitl_checkpoint_node
from src.hitl import queue
from src.models.schemas import PolicyCheckOutput, CoverageStatus
from src.models.state import ClaimsState, initial_state


@pytest.fixture
def isolated_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "DB_PATH", tmp_path / "hitl.db")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit"))
    return tmp_path / "hitl.db"


def _low_value_claim():
    return {
        "claim_id": "CLM-HITL-1", "policy_number": "POL-X", "claimant_name": "A",
        "claimant_email": "a@example.com", "claimant_phone": "1", "claimant_dob": "1990-01-01",
        "incident_date": "2026-01-01", "incident_type": "auto_collision",
        "incident_description": "minor dent", "incident_location": "", "police_report_number": "PR1",
        "estimated_amount": 900.0, "vehicle_year": 2020, "vehicle_make": "X", "vehicle_model": "Y",
        "documents": [], "is_appeal": False, "original_claim_id": None,
    }


def _graph():
    g = StateGraph(ClaimsState)
    g.add_node("hitl_after_policy", hitl_checkpoint_node)
    g.add_edge(START, "hitl_after_policy")
    g.add_edge("hitl_after_policy", END)
    return g.compile(checkpointer=MemorySaver())


def _ticket_count(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM hitl_queue").fetchone()[0]


def test_gate_pause_happens_even_when_generic_triggers_are_quiet(isolated_queue):
    # Low value, no fraud output: check_hitl_required alone finds nothing.
    # The node must still pause because the policy gate sent it here.
    state = initial_state(_low_value_claim())
    state["execution_start_time"] = "run-1"
    state["policy_output"] = PolicyCheckOutput(
        coverage_status=CoverageStatus.NEEDS_VERIFICATION, covered_amount_usd=0, deductible_usd=0,
        exclusions_triggered=[], coverage_notes="unclear", compliance_flags=[], policy_limits={},
        confidence=0.40,
    )
    app, cfg = _graph(), {"configurable": {"thread_id": "t1"}}
    out = app.invoke(state, cfg)
    assert out.get("__interrupt__"), "the gate node must pause"
    payload = out["__interrupt__"][0].value
    assert payload["node"] == "hitl_after_policy"
    assert "policy_checker confidence 0.40" in payload["triggers"][0]


def test_resume_does_not_create_a_second_ticket(isolated_queue):
    state = initial_state(_low_value_claim())
    state["execution_start_time"] = "run-2"
    state["policy_output"] = PolicyCheckOutput(
        coverage_status=CoverageStatus.NEEDS_VERIFICATION, covered_amount_usd=0, deductible_usd=0,
        exclusions_triggered=[], coverage_notes="unclear", compliance_flags=[], policy_limits={},
        confidence=0.40,
    )
    app, cfg = _graph(), {"configurable": {"thread_id": "t2"}}
    paused = app.invoke(state, cfg)
    ticket = paused["__interrupt__"][0].value["ticket_id"]
    assert _ticket_count(isolated_queue) == 1

    final = app.invoke(Command(resume={"decision": "approved", "reviewer_id": "rev", "notes": "ok"}), cfg)
    assert _ticket_count(isolated_queue) == 1
    assert final["hitl_ticket_id"] == ticket
    assert final["human_decision"] == "approved"


def test_ticket_triggers_do_not_repeat_the_fraud_reason(isolated_queue):
    from src.models.schemas import FraudAssessmentOutput, FraudRiskLevel
    state = initial_state({**_low_value_claim(), "claim_id": "CLM-HITL-3"})
    state["execution_start_time"] = "run-3"
    state["fraud_output"] = FraudAssessmentOutput(
        fraud_risk_level=FraudRiskLevel.HIGH, fraud_score=0.8, primary_concerns=[], recommendation="escalate",
        crew_summary="x", pattern_score=0.9, anomaly_score=0.4, consistency_score=0.1, narrative_risk=0.9,
        confidence=0.85,
    )
    g = StateGraph(ClaimsState)
    g.add_node("hitl_checkpoint", hitl_checkpoint_node)
    g.add_edge(START, "hitl_checkpoint")
    g.add_edge("hitl_checkpoint", END)
    out = g.compile(checkpointer=MemorySaver()).invoke(state, {"configurable": {"thread_id": "t3"}})
    triggers = out["__interrupt__"][0].value["triggers"]
    assert sum(t.startswith("Fraud score") for t in triggers) == 1

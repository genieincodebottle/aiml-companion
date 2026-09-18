"""Routing contracts for src/agents/graph.py.

Covers the fixes to route_after_fraud (config-driven auto-reject, a real crew
confidence for the fraud gate), guardrail halts in every router, and where a
claim goes after a human decision.
"""

from types import SimpleNamespace

import pytest

from src.agents import graph
from src.models.schemas import FraudRiskLevel


def fraud(score, level=FraudRiskLevel.LOW, confidence=0.85):
    return SimpleNamespace(fraud_score=score, fraud_risk_level=level, confidence=confidence,
                           narrative_risk=None, primary_concerns=[])


def test_confirmed_fraud_is_auto_rejected():
    state = {"fraud_output": fraud(0.81, FraudRiskLevel.CONFIRMED)}
    assert graph.route_after_fraud(state) == "auto_reject"


@pytest.mark.parametrize("score, expected", [
    (0.44, "damage_assessor"),
    (0.45, "hitl_checkpoint"),   # hitl.triggers.fraud_score in base.yaml
    (0.80, "hitl_checkpoint"),   # high but not CONFIRMED: a human decides
])
def test_review_threshold_comes_from_hitl_config(score, expected):
    state = {"fraud_output": fraud(score, FraudRiskLevel.HIGH if score >= 0.65 else FraudRiskLevel.LOW)}
    assert graph.route_after_fraud(state) == expected


def test_fraud_gate_fires_when_the_crew_does_not_trust_its_own_assessment():
    # Used to read consistency_score, hardcoded to 0.5 against a 0.50 gate
    # with a strict `<`: this branch could never fire.
    state = {"fraud_output": fraud(0.20, confidence=0.45)}
    assert graph.route_after_fraud(state) == "hitl_after_fraud"


@pytest.mark.parametrize("router", [
    graph.route_after_intake,
    graph.route_after_fraud,
    graph.route_after_damage,
    graph.route_after_policy,
    graph.route_after_settlement,
    graph.route_after_evaluation,
])
def test_every_router_sends_a_halted_claim_to_a_human(router):
    assert router({"guardrails_halted": True, "claim": {"estimated_amount": 100}}) == "hitl_checkpoint"


@pytest.mark.parametrize("state, expected", [
    ({"guardrails_halted": True, "human_decision": "approved"}, "communication_agent"),
    ({"human_decision": "denied", "damage_output": None}, "communication_agent"),
    ({"human_decision": "fraud_investigation", "damage_output": None}, "communication_agent"),
    ({"human_decision": "approved", "damage_output": None}, "damage_assessor"),
    ({"human_decision": "approved", "damage_output": object()}, "communication_agent"),
])
def test_after_a_human_decision(state, expected):
    assert graph.route_after_hitl_checkpoint(state) == expected


def test_a_confidence_gate_node_always_has_a_reason_to_pause():
    state = {"policy_output": SimpleNamespace(confidence=0.45)}
    reasons = graph._routing_reasons("hitl_after_policy", state)
    assert reasons and "policy_checker confidence 0.45 below 0.60" in reasons[0]


def test_hitl_checkpoint_reasons_name_the_failed_quality_gate():
    state = {"evaluation_passed": False, "evaluation_output": SimpleNamespace(overall_score=0.52)}
    reasons = graph._routing_reasons("hitl_checkpoint", state)
    assert any("quality gate failed (score 0.52" in r for r in reasons)


def _eval_state(amount, **extra):
    state = {"claim": {"estimated_amount": amount}, "evaluation_passed": True}
    state.update(extra)
    return state


def test_high_value_claims_pause_for_settlement_signoff():
    # US hitl.triggers.min_amount is $10,000. The README always promised this;
    # before, the amount rule only ran inside a HITL node nobody routed to.
    assert graph.route_after_evaluation(_eval_state(15_000)) == "hitl_checkpoint"
    assert graph.route_after_evaluation(_eval_state(9_000)) == "communication_agent"


def test_a_claim_a_human_already_reviewed_is_not_paused_twice_for_value():
    assert graph.route_after_evaluation(_eval_state(15_000, hitl_required=True)) == "communication_agent"


def test_signoff_reason_names_the_threshold():
    state = _eval_state(15_000, settlement_output=object())
    assert any("needs human sign-off" in r for r in graph._routing_reasons("hitl_checkpoint", state))

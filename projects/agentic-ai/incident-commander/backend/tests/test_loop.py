"""End-to-end tests for the Incident Commander loop.

The application is pure-Gemini (no deterministic fallback). These tests inject a
deterministic `FakeLLMClient` in place of the real model so the loop's control flow
can be exercised for free and without flakiness. The stub is test-only; it is never
part of the running app.

This file covers the happy paths and the overall shape of a run. Halt conditions
live in `test_halt_conditions.py`, safety in `test_guardrails.py`, the evidence
branch in `test_evidence_loop.py`, and the HTTP layer in `test_api.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from src.models.state import ActionStatus, AgentName, EventType, HaltReason, IncidentStatus
from src.sandbox.scenarios import SCENARIOS, list_scenarios

from .conftest import drive


def test_approve_path_resolves(manager):
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.RESOLVED
    assert state.halt_reason == HaltReason.RECOVERED
    assert state.iterations >= 1
    assert state.postmortem is not None
    assert any(a.result and a.result.get("effective") for a in state.actions)


def test_reject_path_escalates(manager):
    state = asyncio.run(drive(manager, "memory_leak", "reject"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.HUMAN_REJECTED
    assert state.postmortem is not None  # a postmortem is written on every outcome
    assert state.actions[-1].status == ActionStatus.REJECTED


def test_compound_fault_requires_multiple_iterations(manager):
    # cascading_failure needs an ordered 2-step remediation, so the loop must
    # iterate and re-diagnose rather than resolve in a single pass.
    state = asyncio.run(drive(manager, "cascading_failure", "approve"))
    assert state.status == IncidentStatus.RESOLVED
    assert state.iterations >= 2, "compound fault should take at least two loop passes"
    assert any(a.result and a.result.get("progress") and not a.result.get("effective") for a in state.actions), (
        "the first step should make partial progress without clearing the fault"
    )
    assert any(a.result and a.result.get("effective") for a in state.actions)
    # It must actually RE-diagnose, not just restate the same conclusion.
    causes = [d["leading_cause"] for d in state.diagnostics]
    assert len(set(causes)) > 1, "the second pass should reach a different leading cause"


@pytest.mark.parametrize("scenario_id", [s["id"] for s in list_scenarios()])
def test_every_scenario_reaches_its_expected_outcome(manager, scenario_id):
    """Including the negative ones: escalating and doing nothing are correct answers."""
    state = asyncio.run(drive(manager, scenario_id, "approve"))
    expected = SCENARIOS[scenario_id].expected_outcome
    actual = "resolved" if state.status == IncidentStatus.RESOLVED else state.status.value
    assert actual == expected, f"{scenario_id}: expected {expected}, got {actual}"


def test_unfixable_fault_escalates_rather_than_flailing(manager):
    """An expired third-party certificate cannot be fixed from here.

    The valuable behaviour is recognizing that and handing over - NOT cycling
    through every action hoping one sticks. An agent that "resolves" this has
    caused disruption and still not fixed anything.
    """
    state = asyncio.run(drive(manager, "unfixable_ca_expiry", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    executed = [a.action_type for a in state.actions if a.status == ActionStatus.EXECUTED]
    assert all(a == "no_op" for a in executed), f"should not have taken disruptive actions, took {executed}"


def test_false_alarm_takes_no_action(manager):
    """A healthy service must not be 'fixed'. Restarting it would BE the outage."""
    state = asyncio.run(drive(manager, "false_alarm", "approve"))
    assert state.status == IncidentStatus.RESOLVED
    disruptive = [a.action_type for a in state.actions if a.action_type != "no_op"]
    assert not disruptive, f"acted on a healthy service: {disruptive}"
    assert not any((a.result or {}).get("harmful") for a in state.actions)


def test_expected_signal_is_predicted_and_then_checked(manager):
    """The falsification loop: the proposal predicts, the verifier judges the prediction."""
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.actions[0].expected_signal, "the proposal must make a checkable prediction"
    assert state.verification_history, "the verifier must run"
    verification = state.verification_history[-1]
    assert "expected_signal_met" in verification
    assert verification["signal_assessment"], "the verifier must explain the comparison"


def test_critic_reviews_every_proposal(manager):
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    action = state.actions[0]
    assert action.critique is not None, "the Critic should review before a human is asked"
    # A critic that is allowed to say nothing is worth nothing.
    assert action.critique["strongest_objection"]


def test_postmortem_is_written_on_every_outcome(make_manager):
    for scenario, decision in [("bad_deploy", "approve"), ("memory_leak", "reject")]:
        state = asyncio.run(drive(make_manager(), scenario, decision))
        assert state.postmortem is not None
        assert state.postmortem["runbook_lesson"], "the lesson feeds runbook memory; it must not be empty"
        assert state.postmortem["halt_reason"]


def test_events_are_sequenced_and_replayable(manager):
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    seqs = [e.seq for e in state.events]
    assert seqs == sorted(seqs), "event sequence numbers must be monotonic"
    assert seqs == list(range(1, len(seqs) + 1)), "sequence must be gapless for Last-Event-ID resume"


def test_run_emits_the_expected_agent_participation(manager):
    """Every specialist should appear in the audit trail.

    The Supervisor is checked only on the explicit-loop engine: the graph engine
    routes through `StateGraph` conditional edges (pure functions of state) rather
    than through an LLM router, which is a documented architectural difference
    between the two, not a gap. See `graph.py` and ADR-0002.
    """
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    agents = {e.agent for e in state.events if e.agent}
    if state.engine == "loop":
        assert AgentName.SUPERVISOR in agents, "the LLM Supervisor never ran"
    for expected in (
        AgentName.TRIAGE,
        AgentName.DIAGNOSTICIAN,
        AgentName.REMEDIATION,
        AgentName.CRITIC,
        AgentName.VERIFIER,
        AgentName.POSTMORTEM,
    ):
        assert expected in agents, f"{expected.value} never ran"


def test_risk_is_assigned_by_guardrails_not_by_the_model(manager):
    """The model reports its own view of risk; the guardrail layer overrides it.

    An agent that could grade its own rollback "low risk" could route around the
    approval gate, so this is a security property, not a style preference.
    """
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    action = state.actions[0]
    assert action.action_type == "rollback_deploy"
    assert action.risk.value == "high", "rollback is HIGH regardless of what the model claimed"


def test_high_risk_action_always_stops_for_a_human_even_in_auto_approve(manager):
    """auto_approve is a demo convenience; it must never cover a high-risk action."""
    saw_gate = False

    def watch(ev, mgr, st):
        nonlocal saw_gate
        if ev.type == EventType.AWAITING_APPROVAL:
            saw_gate = True

    state = asyncio.run(drive(manager, "bad_deploy", "approve", auto_approve=True, on_event=watch))
    assert saw_gate, "a rollback must stop for a human even with auto-approve on"
    assert state.status == IncidentStatus.RESOLVED

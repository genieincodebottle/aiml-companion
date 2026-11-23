"""Parity between the two engines.

The project ships the same agent loop twice: `loop.py` (an explicit `while`, the
teaching reference) and `graph.py` (a LangGraph `StateGraph` with a checkpointer,
the production shape). That is only a useful lesson if they are genuinely the same
system - two implementations that quietly diverge teach the wrong thing, and the
divergence would show up first as an inconsistent demo.

These tests are the contract that keeps them honest. They also pin the routing edge
functions directly, which is the ergonomic advantage of the graph formulation: the
branch conditions are pure functions of state and can be tested without running a
graph at all.
"""

from __future__ import annotations

import asyncio

import pytest

from src.graph import (
    route_after_critique,
    route_after_diagnose,
    route_after_gate,
    route_after_guard,
    route_after_observe,
    route_after_verify,
    run_incident,
)
from src.guardrails.manager import GuardrailManager
from src.models.state import (
    ActionStatus,
    HaltReason,
    IncidentState,
    IncidentStatus,
    RemediationAction,
    Severity,
)
from src.sandbox.scenarios import SCENARIOS, list_scenarios
from src.sandbox.stack import SandboxStack

from ._fake_llm import FakeLLMClient
from .conftest import drive

SCENARIO_IDS = [s["id"] for s in list_scenarios()]


def _run_graph(scenario_id: str, approve: bool = True) -> IncidentState:
    scenario = SCENARIOS[scenario_id]
    state = IncidentState(
        scenario_id=scenario.id,
        service=scenario.default_service,
        severity=Severity(scenario.severity),
        engine="graph",
    )
    stack = SandboxStack(scenario=scenario, service=scenario.default_service)
    guard = GuardrailManager()
    return run_incident(state, stack, guard, FakeLLMClient(), lambda a: (approve, "test"))


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_both_engines_reach_the_same_outcome(scenario_id, make_manager):
    """The headline contract: same agents, same guardrails, same verdict."""
    loop_state = asyncio.run(drive(make_manager(), scenario_id, "approve"))
    graph_state = _run_graph(scenario_id)
    assert graph_state.status == loop_state.status, (
        f"{scenario_id}: loop said {loop_state.status.value}, graph said {graph_state.status.value}"
    )


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_both_engines_match_the_expected_outcome(scenario_id):
    graph_state = _run_graph(scenario_id)
    expected = SCENARIOS[scenario_id].expected_outcome
    actual = "resolved" if graph_state.status == IncidentStatus.RESOLVED else graph_state.status.value
    assert actual == expected


def test_graph_honours_the_human_gate():
    """`interrupt_before=['act']` means the graph CHECKPOINTS at the gate.

    That is the one thing the graph genuinely does better than the explicit loop:
    the pending approval lives in the checkpointer, not in one process's memory, so
    it can be answered later, from elsewhere, after a restart.
    """
    state = _run_graph("bad_deploy", approve=False)
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.HUMAN_REJECTED
    assert state.actions[-1].status == ActionStatus.REJECTED
    assert not any((a.result or {}).get("applied") for a in state.actions), "nothing may run unapproved"


def test_graph_applies_the_same_guardrail_layer():
    """Risk classification and RBAC are not re-implemented per engine."""
    state = _run_graph("bad_deploy")
    assert state.actions[0].risk.value == "high"


def test_graph_resists_injection_identically():
    state = _run_graph("poisoned_telemetry")
    actions = [a.action_type for a in state.actions]
    assert "failover" not in actions
    assert any(f["kind"] == "injection" for f in state.guardrail_flags)


def test_graph_runs_the_evidence_loop():
    state = _run_graph("silent_config_drift")
    assert state.evidence_rounds >= 1
    assert state.status == IncidentStatus.RESOLVED


# --- the routing edges, as pure functions -----------------------------------


def _state(**kw) -> IncidentState:
    return IncidentState(scenario_id="bad_deploy", service="svc", **kw)


def test_route_after_guard_sends_a_halted_incident_to_finish():
    state = _state(status=IncidentStatus.ESCALATED)
    assert route_after_guard({"incident": state}) == "finish"
    assert route_after_guard({"incident": _state()}) == "observe"


def test_route_after_observe_triages_once_then_diagnoses():
    assert route_after_observe({"incident": _state()}) == "triage"
    assert route_after_observe({"incident": _state(triage_summary="done")}) == "diagnose"


class _Diagnosis:
    def __init__(self, confidence, needs_more_data=False, requested_tools=()):
        self.confidence = confidence
        self.needs_more_data = needs_more_data
        self.requested_tools = list(requested_tools)


def test_route_after_diagnose_gathers_evidence_when_unsure_and_tools_were_named():
    from src.models.state import TelemetryTool

    state = _state(triage_summary="x")
    diagnosis = _Diagnosis(0.3, needs_more_data=True, requested_tools=[TelemetryTool.CONFIG_DIFF])
    assert route_after_diagnose({"incident": state, "diagnosis": diagnosis}) == "gather_evidence"


def test_route_after_diagnose_escalates_rather_than_acting_on_a_guess():
    """No tools left to try and still unsure: handing over is the correct move."""
    state = _state(triage_summary="x", evidence_rounds=99)
    diagnosis = _Diagnosis(0.2, needs_more_data=True, requested_tools=[])
    assert route_after_diagnose({"incident": state, "diagnosis": diagnosis}) == "finish"
    assert state.halt_reason == HaltReason.LOW_CONFIDENCE


def test_route_after_diagnose_proposes_when_confident():
    state = _state(triage_summary="x")
    assert route_after_diagnose({"incident": state, "diagnosis": _Diagnosis(0.9)}) == "propose"


def test_route_after_critique_loops_back_on_a_veto():
    state = _state()
    state.actions.append(RemediationAction(action_type="failover", target="svc", status=ActionStatus.REJECTED))
    assert route_after_critique({"incident": state}) == "guard"


def test_route_after_critique_proceeds_on_an_endorsement():
    state = _state()
    state.actions.append(RemediationAction(action_type="restart_service", target="svc"))
    assert route_after_critique({"incident": state}) == "gate"


def test_route_after_gate_and_verify():
    assert route_after_gate({"incident": _state(status=IncidentStatus.ESCALATED)}) == "finish"
    assert route_after_gate({"incident": _state(status=IncidentStatus.REMEDIATING)}) == "act"
    assert route_after_verify({"incident": _state(status=IncidentStatus.RESOLVED)}) == "finish"
    assert route_after_verify({"incident": _state(status=IncidentStatus.VERIFYING)}) == "guard"


def test_a_repeat_block_goes_round_again_instead_of_escalating():
    """Same as the loop: a repeated dud is recoverable; only an RBAC block escalates."""
    state = _state(status=IncidentStatus.DIAGNOSING)
    state.actions.append(RemediationAction(action_type="restart_service", target="svc", status=ActionStatus.BLOCKED))
    assert route_after_gate({"incident": state}) == "guard"

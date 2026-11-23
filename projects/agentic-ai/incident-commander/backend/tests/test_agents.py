"""Agent-level tests: prompt construction, routing policy, and the Critic.

These test the pieces that are pure functions of state, which is most of the
interesting logic. The prompt-construction tests in particular are worth having:
the fencing and separation rules in `_prompting.py` are the *only* thing standing
between an attacker-controlled log line and the model's instruction channel, and a
refactor that quietly drops the fence would otherwise be invisible.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents import _prompting, critic, diagnostician, remediation, supervisor, triage, verifier
from src.agents.supervisor import ROUTES, default_route, sanitize_route
from src.loop import IncidentManager
from src.models.schemas import CriticVerdict
from src.models.state import (
    ActionRisk,
    ActionStatus,
    ActionType,
    HaltReason,
    IncidentState,
    IncidentStatus,
    RemediationAction,
    TelemetryTool,
)

from ._fake_llm import FakeLLMClient, ScriptedLLMClient
from .conftest import drive


def _state(**kw) -> IncidentState:
    return IncidentState(scenario_id="bad_deploy", service="api-gateway", **kw)


# --- prompt construction ----------------------------------------------------


def test_telemetry_is_always_fenced_as_untrusted():
    """The fence is the only thing separating log text from the instruction channel."""
    block = _prompting.telemetry_block({"logs": ["something"]})
    assert "UNTRUSTED_TELEMETRY" in block
    assert "END_UNTRUSTED_TELEMETRY" in block


def test_trusted_context_is_kept_out_of_the_untrusted_fence():
    """Mixing our own facts with attacker-influenced text lends injected content false provenance."""
    state = _state(triage_summary="checkout is failing")
    context = _prompting.incident_context(state)
    assert "TRUSTED" in context
    assert "UNTRUSTED_TELEMETRY" not in context
    # The triage summary is model-written from telemetry: fenced, never "trusted".
    assert "checkout is failing" not in context
    assert "UNTRUSTED_TELEMETRY" in _prompting.triage_block(state)


def test_every_system_prompt_carries_the_standing_rules():
    for agent in (triage, diagnostician, remediation, critic, verifier, supervisor):
        prompt = agent._SYSTEM  # noqa: SLF001 - inspecting the prompt is the point
        assert "UNTRUSTED_TELEMETRY" in prompt, f"{agent.__name__} does not warn about untrusted data"
        assert "never execute" in prompt.lower() or "route only" in prompt.lower()


def test_action_history_reports_outcomes_not_just_names():
    """ "restart was tried" and "restart was tried and did nothing" imply different moves."""
    state = _state()
    state.actions.append(
        RemediationAction(
            action_type="restart_service",
            target="svc",
            status=ActionStatus.EXECUTED,
            result={"effective": False, "progress": False, "reason": "no change"},
            iteration=1,
        )
    )
    block = _prompting.action_history_block(state)
    assert "NO EFFECT" in block
    assert "do not repeat" in block.lower()


def test_memory_is_framed_as_prior_art_not_ground_truth():
    """Framing matters: presented as fact, a recalled case anchors the model."""
    state = _state()
    state.recalled_episodes = [{"score": 0.9, "prompt_line": "[RESOLVED] symptom: heap exhausted"}]
    block = _prompting.memory_block(state)
    assert "NOT ground truth" in block
    assert "trust today" in block.lower()


def test_tool_catalog_is_rendered_into_the_diagnostician_prompt():
    """The agent cannot request a tool it was never told about."""
    assert TelemetryTool.CONFIG_DIFF.value in diagnostician._SYSTEM  # noqa: SLF001
    assert TelemetryTool.POOL_STATS.value in diagnostician._SYSTEM  # noqa: SLF001


def test_prompt_truncation_bounds_a_runaway_log_tail():
    """One enormous log tail must not crowd the task out of the context window."""
    huge = {"logs": ["x" * 50_000]}
    block = _prompting.telemetry_block(huge)
    assert "truncated" in block
    assert len(block) < 10_000


# --- routing policy ---------------------------------------------------------


def test_default_route_triages_first_then_diagnoses():
    state = _state(iterations=1)
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "triage"
    state.triage_summary = "done"
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "diagnose"


def test_default_route_gathers_evidence_when_unsure():
    state = _state(triage_summary="x")
    state.diagnostics.append({"confidence": 0.3, "needs_more_data": True, "requested_tools": ["get_config_diff"]})
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "gather_evidence"


def test_default_route_escalates_when_evidence_is_exhausted():
    state = _state(triage_summary="x", evidence_rounds=2)
    state.diagnostics.append({"confidence": 0.3, "needs_more_data": True, "requested_tools": []})
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "escalate"


def test_default_route_rediagnoses_after_a_verification():
    """The bug this rule prevents: proposing from a diagnosis the telemetry outdated.

    On a compound fault the second step is a *different* cause. An agent that
    re-proposes from a stale diagnosis re-applies the fix it already applied.
    """
    state = _state(triage_summary="x")
    state.diagnostics.append({"confidence": 0.9, "needs_more_data": False, "requested_tools": []})
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "propose"
    state.verification_history.append({"recovered": False})
    assert default_route(state, min_confidence=0.6, max_evidence_rounds=2) == "diagnose"


@pytest.mark.parametrize("route", ROUTES)
def test_sanitize_route_always_returns_a_legal_route(route):
    """The Supervisor is an LLM: it will occasionally route somewhere impossible."""
    state = _state(triage_summary="x")
    state.diagnostics.append({"confidence": 0.9, "needs_more_data": False, "requested_tools": []})
    assert sanitize_route(route, state, min_confidence=0.6, max_evidence_rounds=2) in ROUTES


def test_sanitize_route_refuses_to_act_on_a_low_confidence_diagnosis():
    """LLM proposes, code disposes - applied to control flow."""
    state = _state(triage_summary="x", evidence_rounds=5)
    state.diagnostics.append({"confidence": 0.1, "needs_more_data": True, "requested_tools": []})
    assert sanitize_route("propose", state, min_confidence=0.6, max_evidence_rounds=2) == "escalate"


def test_sanitize_route_repairs_nonsense():
    state = _state()
    assert sanitize_route("delete_production", state, min_confidence=0.6, max_evidence_rounds=2) in ROUTES
    # 'verify' with nothing done yet, and 'resolve' with no confirmed recovery.
    assert sanitize_route("verify", state, min_confidence=0.6, max_evidence_rounds=2) != "verify"
    assert sanitize_route("resolve", state, min_confidence=0.6, max_evidence_rounds=2) != "resolve"


@pytest.mark.loop_engine
def test_supervisor_can_be_replaced_by_the_deterministic_policy(env):
    """Turning the LLM router off and diffing behaviour is a supported experiment."""
    env(IC_SUPERVISOR_ENABLED="false")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.RESOLVED
    route_events = [e for e in state.events if e.type.value == "route"]
    assert route_events and all(e.payload["llm"] is False for e in route_events)


# --- the Critic -------------------------------------------------------------


@pytest.mark.loop_engine
def test_critic_veto_sends_the_loop_back_instead_of_executing():
    """A veto must block the action, not merely be recorded next to it."""
    veto = CriticVerdict(
        endorse=False,
        concerns=["the evidence does not support a rollback"],
        strongest_objection="No deploy correlates with the error onset.",
        safer_alternative=ActionType.RESTART_SERVICE,
        confidence=0.9,
    )
    manager = IncidentManager(llm=ScriptedLLMClient({CriticVerdict: veto}))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.CRITIC_VETO
    assert not any((a.result or {}).get("applied") for a in state.actions), "a vetoed action must not run"


@pytest.mark.loop_engine
def test_repeated_critic_vetoes_are_bounded(env):
    """A stubborn Critic and a stubborn Planner must not argue until the budget is gone."""
    env(IC_MAX_CRITIC_VETOES="1")
    veto = CriticVerdict(endorse=False, concerns=["no"], strongest_objection="Insufficient evidence.", confidence=0.9)
    manager = IncidentManager(llm=ScriptedLLMClient({CriticVerdict: veto}))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.halt_reason == HaltReason.CRITIC_VETO
    assert state.iterations <= 3


def test_critic_can_be_disabled_for_comparison(env):
    env(IC_CRITIC_ENABLED="false")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.RESOLVED
    assert all(a.critique is None for a in state.actions)


def test_critic_blocks_acting_on_a_healthy_service():
    """The one case where a veto is unambiguously correct."""
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "false_alarm", "approve"))
    assert not any(a.action_type != "no_op" for a in state.actions)


# --- agent contracts --------------------------------------------------------


def test_agents_never_mutate_state():
    """Agents read and return; only the orchestrator writes. This keeps both engines
    able to drive the same functions without duplicating bookkeeping."""
    state = _state(triage_summary="x")
    before = state.model_dump(mode="json")
    llm = FakeLLMClient()
    observations = {"metrics": {"error_rate": 0.18, "p99_latency_ms": 4200}, "recent_logs": ["release v2.8.1"]}
    diagnostician.run(state, observations, llm)
    triage.run(state, observations, llm)
    assert state.model_dump(mode="json") == before


def test_remediation_output_is_constrained_to_the_action_menu():
    """`action_type` is an enum, so 'hallucinate a new action' is not a failure mode."""
    state = _state(triage_summary="x")
    llm = FakeLLMClient()
    observations = {"metrics": {"error_rate": 0.18}, "recent_logs": ["NullPointerException after release v2.8.1"]}
    diagnosis = diagnostician.run(state, observations, llm)
    proposal = remediation.run(state, diagnosis, observations, llm)
    assert isinstance(proposal.action_type, ActionType)


def test_verifier_receives_the_prediction_it_is_meant_to_check():
    state = _state(triage_summary="x")
    action = RemediationAction(
        action_type="rollback_deploy",
        target="svc",
        risk=ActionRisk.HIGH,
        expected_signal="error_rate drops below 0.01",
    )
    captured = {}

    class Capturing(FakeLLMClient):
        def structured(self, schema, system, user, *a, **kw):
            captured["user"] = user
            return super().structured(schema, system, user, *a, **kw)

    verifier.run(state, {"metrics": {"error_rate": 0.001}}, Capturing(), action)
    assert "error_rate drops below 0.01" in captured["user"], "the verifier must see the claim it is judging"


def test_the_critic_sees_the_evidence_the_diagnostician_gathered():
    """A Critic that cannot see gathered evidence vetoes a correct diagnosis as invented.

    A live run did exactly that on Silent Config Drift: the config diff was in
    state.evidence, the Diagnostician and Remediation saw it, the Critic did not,
    and it vetoed the right fix three times as "hallucinated".
    """
    from src.models.schemas import DiagnosisResult, RemediationProposal
    from src.models.state import ActionRisk, ActionType

    class Capture(FakeLLMClient):
        def structured(self, schema, system, user, *args, **kwargs):
            self.last_user = user
            return super().structured(schema, system, user, *args, **kwargs)

    state = _state(triage_summary="pool starved")
    state.evidence["iter1:get_config_diff"] = [{"key": "db.pool.maxSize", "was": 50, "now": 10}]
    diagnosis = DiagnosisResult(
        hypotheses=[], leading_cause="pool cut from 50 to 10", confidence=0.9, evidence=["config diff"]
    )
    proposal = RemediationProposal(
        action_type=ActionType.SCALE_OUT,
        target="svc",
        risk=ActionRisk.MEDIUM,
        rationale="restore capacity",
        expected_signal="p99 below 500",
    )
    llm = Capture()
    critic.run(state, diagnosis, proposal, ActionRisk.MEDIUM, {"metrics": {}}, llm)
    assert "db.pool.maxSize" in llm.last_user
    assert "ADDITIONAL EVIDENCE YOU PREVIOUSLY REQUESTED" in llm.last_user

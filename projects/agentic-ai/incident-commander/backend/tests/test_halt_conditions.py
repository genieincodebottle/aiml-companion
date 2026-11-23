"""Halt conditions - the tests that matter most for an autonomous system.

An agent loop without provable halt conditions is not a system, it is a liability.
Every ceiling in `config.py` gets a test here that proves the loop actually STOPS,
not merely that the number is stored somewhere.

The original version of this project asserted `budget_exhausted()` arithmetic and
called that a ceiling test. It was not: it proved the comparison operator worked,
never that the loop honoured it. These tests drive real incidents to a stop.
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm import LLMBudgetError, LLMConfigError, LLMUnavailableError
from src.loop import IncidentManager
from src.models.state import HaltReason, IncidentState, IncidentStatus

from ._fake_llm import FailingLLMClient, FakeLLMClient
from .conftest import drive


def test_iteration_ceiling_stops_the_loop(env):
    """A fault that never clears must stop at the ceiling, not run forever."""
    env(IC_MAX_ITERATIONS="2", IC_SUPERVISOR_ENABLED="false")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "unfixable_ca_expiry", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.iterations <= 2, "the loop ran past its iteration ceiling"
    assert state.halt_reason in {
        HaltReason.ITERATION_CEILING,
        HaltReason.VERIFIER_ESCALATED,
        HaltReason.LOW_CONFIDENCE,
    }


def test_cost_ceiling_escalates_before_spending_more(env):
    """A zero budget must stop the incident at the first model call, not after it."""
    env(IC_MAX_COST_USD="0.0")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.COST_CEILING
    assert not state.actions, "no action should be taken once the budget is exhausted"


def test_wallclock_ceiling_is_a_halt_condition(env):
    """An agent that is neither expensive nor iterating can still hang on a slow tool."""
    env(IC_MAX_WALLCLOCK_S="0.0")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.WALLCLOCK_CEILING


def test_approval_timeout_escalates_instead_of_hanging_forever(env):
    """The bug this test exists to prevent.

    A bare `await` on the approval future means an operator who closes their laptop
    strands the incident, its task, and its sandbox in memory permanently. Timing
    out and escalating is both the safe answer and the honest one: nobody answered,
    so nobody approved.
    """
    env(IC_APPROVAL_TIMEOUT_S="0.2")
    manager = IncidentManager(llm=FakeLLMClient())
    # decision=None: never answer the gate.
    state = asyncio.run(asyncio.wait_for(drive(manager, "bad_deploy", None), timeout=15))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.APPROVAL_TIMEOUT
    assert not any(a.result for a in state.actions), "nothing may execute without an approval"


def test_operator_cancel_stops_the_run_and_still_writes_a_postmortem():
    """A stop button that loses the audit trail is not a stop button."""
    manager = IncidentManager(llm=FakeLLMClient())

    def cancel_at_gate(ev, mgr, st):
        if ev.type.value == "awaiting_approval":
            mgr.request_cancel(st.incident_id)

    state = asyncio.run(drive(manager, "bad_deploy", None, on_event=cancel_at_gate))
    assert state.status == IncidentStatus.CANCELLED
    assert state.halt_reason == HaltReason.OPERATOR_CANCELLED
    assert state.postmortem is not None, "a cancelled incident still needs a postmortem"


def test_low_confidence_escalates_rather_than_guessing(env):
    """Refusing to act on a diagnosis the diagnostician does not believe."""
    env(IC_MIN_ACTION_CONFIDENCE="0.99", IC_MAX_EVIDENCE_ROUNDS="0")
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.LOW_CONFIDENCE
    assert not state.actions, "an unsure agent must not act"


def test_model_outage_escalates_and_does_not_crash():
    """When the model is down the correct behaviour is to hand over, not to explode."""
    manager = IncidentManager(llm=FailingLLMClient(LLMUnavailableError("model is down")))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.LLM_UNAVAILABLE


@pytest.mark.parametrize("engine", ["loop", "graph"])
def test_outage_postmortem_does_not_ask_the_failed_model_again(env, engine):
    """Found live with an invalid key: the writeup retried the dead model, so the record
    arrived seconds after the terminal status and the extra failures fed the breaker."""
    env(IC_ENGINE=engine)
    llm = FailingLLMClient(LLMUnavailableError("model is down"))
    state = asyncio.run(drive(IncidentManager(llm=llm), "bad_deploy", "approve"))
    assert state.halt_reason == HaltReason.LLM_UNAVAILABLE
    assert llm.calls == 1, "only the agent call that failed; the postmortem must not call again"
    assert state.postmortem and "not asked again" in state.postmortem["summary"]


def test_missing_api_key_fails_cleanly_rather_than_silently_degrading():
    """No key means no incident. There is deliberately no non-agentic fallback path."""
    manager = IncidentManager(llm=FailingLLMClient(LLMConfigError("GOOGLE_API_KEY is not configured")))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.FAILED
    assert state.halt_reason == HaltReason.LLM_UNAVAILABLE
    assert any(e.payload.get("kind") == "llm_not_configured" for e in state.events)


def test_budget_error_mid_run_escalates():
    manager = IncidentManager(llm=FailingLLMClient(LLMBudgetError("would exceed remaining budget")))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.ESCALATED
    assert state.halt_reason == HaltReason.COST_CEILING


def test_a_loop_crash_is_surfaced_never_swallowed():
    manager = IncidentManager(llm=FailingLLMClient(ZeroDivisionError("boom")))
    state = asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert state.status == IncidentStatus.FAILED
    assert state.halt_reason == HaltReason.CRASH
    assert any(e.payload.get("kind") == "loop_crash" for e in state.events)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("iterations", 6, HaltReason.ITERATION_CEILING),
        ("total_cost_usd", 1.0, HaltReason.COST_CEILING),
        ("cancel_requested", True, HaltReason.OPERATOR_CANCELLED),
    ],
)
def test_exhausted_reason_names_the_ceiling_that_tripped(field, value, expected):
    """`HaltReason` is an enum, not free text, so the eval can count how runs end."""
    state = IncidentState(scenario_id="bad_deploy", service="svc", max_iterations=6, max_cost_usd=0.5)
    setattr(state, field, value)
    assert state.exhausted_reason() == expected


def test_incident_eviction_bounds_memory_growth(env):
    """Retention is a real leak in any long-running agent service.

    Without eviction, `incidents`, `stacks`, `guards` and the token ledger grow for
    the life of the process - the kind of slow leak that only shows up in the
    deployment nobody load-tests.
    """
    env(IC_MAX_RETAINED_INCIDENTS="3")
    manager = IncidentManager(llm=FakeLLMClient())
    for _ in range(6):
        asyncio.run(drive(manager, "bad_deploy", "approve"))
    assert len(manager.incidents) <= 3
    assert len(manager.stacks) <= 3, "sandboxes must be evicted alongside their incidents"
    assert len(manager.guards) <= 3

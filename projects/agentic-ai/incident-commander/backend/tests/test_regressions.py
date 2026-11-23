"""Regression tests for bugs found by the pre-publication audit.

Each test drives the REAL path (the manager, both engines) and was red before its
fix. They are kept together so a reader can see what went wrong and how it is
pinned down: several are exactly the "control that was documented and tested for
existence, but never reached on the real path" class of bug.
"""

from __future__ import annotations

import asyncio

import pytest

from evaluation.metrics import is_correct_outcome
from src.llm import LLMUnavailableError
from src.memory.runbook import Episode, get_memory
from src.models.schemas import CriticVerdict
from src.models.state import EventType, HaltReason, IncidentStatus
from src.sandbox.scenarios import SCENARIOS

from ._fake_llm import FailingLLMClient, FakeLLMClient
from .conftest import drive

ENGINES = ["loop", "graph"]


class RecordingLLM(FakeLLMClient):
    """The offline agent, plus a log of every (agent, prompt) it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[tuple[str, str]] = []

    def structured(self, schema, system, user, *args, **kwargs):
        self.prompts.append((kwargs.get("agent", "unknown"), user))
        return super().structured(schema, system, user, *args, **kwargs)


# --- cancel ------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_cancel_after_the_first_action_stops_the_second(engine, env, make_manager):
    """Cancel is a stop button: nothing may execute after it is pressed.

    Before the fix, the graph engine ignored a cancel (the flag lived on the
    manager's object, the graph ran on its checkpointed copy) and went on to run
    the second action, ending `resolved`.
    """
    env(IC_ENGINE=engine)

    def cancel_after_first_action(ev, manager, state):
        if ev.type == EventType.ACTION_EXECUTED:
            manager.request_cancel(state.incident_id)

    state = asyncio.run(drive(make_manager(), "cascading_failure", "approve", on_event=cancel_after_first_action))
    executed = [a for a in state.actions if (a.result or {}).get("applied")]
    assert state.status == IncidentStatus.CANCELLED
    assert state.halt_reason == HaltReason.OPERATOR_CANCELLED
    assert len(executed) == 1


# --- engine parity on the hard cases ---------------------------------------


def test_both_engines_honour_the_iteration_ceiling_identically(env, make_manager):
    """The graph used to count the pass before checking the ceiling, halting one pass early."""
    results = {}
    for engine in ENGINES:
        env(IC_ENGINE=engine, IC_MAX_ITERATIONS="2")
        state = asyncio.run(drive(make_manager(), "cascading_failure", "approve"))
        results[engine] = (state.status, state.halt_reason)
    assert results["loop"] == results["graph"] == (IncidentStatus.RESOLVED, HaltReason.RECOVERED)


def test_the_graph_engine_writes_exactly_one_postmortem(env):
    env(IC_ENGINE="graph")
    from src.loop import IncidentManager

    llm = FakeLLMClient()
    state = asyncio.run(drive(IncidentManager(llm=llm), "bad_deploy", "approve"))
    assert state.postmortem is not None
    assert [agent for agent, _ in llm.calls].count("Postmortem") == 1


# --- the Critic's objection reaches the next diagnosis ----------------------


def test_a_critic_veto_sends_the_objection_into_the_next_diagnosis(make_manager):
    """A veto must change what the planner sees; otherwise it just re-proposes the same thing."""

    class VetoOnce(RecordingLLM):
        vetoed = False

        def structured(self, schema, system, user, *args, **kwargs):
            if schema is CriticVerdict and not self.vetoed:
                self.vetoed = True
                self.prompts.append((kwargs.get("agent", "unknown"), user))
                return CriticVerdict(
                    endorse=False,
                    concerns=["not convinced"],
                    strongest_objection="OBJECTION-MARKER-42",
                    confidence=0.9,
                )
            return super().structured(schema, system, user, *args, **kwargs)

    llm = VetoOnce()
    state = asyncio.run(drive(make_manager(llm), "bad_deploy", "approve"))
    later_diagnoses = [u for agent, u in llm.prompts if agent == "Diagnostician" and "OBJECTION-MARKER-42" in u]
    assert later_diagnoses, "the re-diagnosis never saw why the proposal was vetoed"
    assert state.status == IncidentStatus.RESOLVED


# --- outages -----------------------------------------------------------------


def _outage_run(make_manager, scenario_id: str = "unfixable_ca_expiry"):
    manager = make_manager(FailingLLMClient(LLMUnavailableError("model is down")))
    return manager, asyncio.run(drive(manager, scenario_id, "approve"))


def test_an_outage_still_gets_a_postmortem(make_manager):
    """Invariant 6: every terminal incident has a postmortem, including this one."""
    _, state = _outage_run(make_manager)
    assert state.halt_reason == HaltReason.LLM_UNAVAILABLE
    assert state.postmortem is not None
    assert "No model wrote this summary" in state.postmortem["summary"]
    assert state.postmortem["runbook_lesson"] == ""  # nothing invented is ever recalled


def test_an_outage_is_not_scored_as_a_correct_escalation(make_manager):
    """A dead model on a must-escalate scenario used to score 100%."""
    _, state = _outage_run(make_manager)
    scenario = SCENARIOS["unfixable_ca_expiry"]
    assert not is_correct_outcome("escalated", state.halt_reason.value, scenario)
    assert is_correct_outcome("escalated", "verifier_escalated", scenario)


def test_an_outage_is_not_written_into_runbook_memory(make_manager):
    _outage_run(make_manager)
    assert len(get_memory()) == 0


def test_a_late_viewer_of_a_failed_incident_gets_a_closed_stream(make_manager):
    """The stream used to stay open forever when the run ended without a postmortem."""
    manager, state = _outage_run(make_manager)

    async def read_all():
        return [ev async for ev in manager.subscribe(state.incident_id)]

    events = asyncio.run(asyncio.wait_for(read_all(), timeout=5))
    assert events, "the replay should still deliver the history"


# --- memory ------------------------------------------------------------------


def test_the_scenario_id_is_not_part_of_the_memory_index():
    """The scenario id is the simulator's answer key; recall must work from symptoms."""
    episode = Episode(
        episode_id="ep_1",
        scenario_id="zzanswerkey",
        service="checkout-api",
        symptom="error rate spiked after release",
        root_cause="bad deploy",
    )
    assert "zzanswerkey" not in episode.symptom_terms


# --- code has the last word on preconditions and on "recovered" -------------


def test_registry_preconditions_are_enforced_at_the_guardrail():
    """These were defined and unit-tested, but nothing on the real path called them."""
    from src.guardrails.manager import GuardrailManager

    guard = GuardrailManager()
    no_alternate = {"dependency_health": [{"provider": "partner-api", "status": "failing"}], "deploys": []}
    blocked = guard.check_action("failover", [], no_alternate)
    assert blocked is not None and blocked.kind == "precondition_block"
    no_recent_deploy = {"dependency_health": [], "deploys": [{"version": "v1", "within_incident_window": False}]}
    assert guard.check_action("rollback_deploy", [], no_recent_deploy).kind == "precondition_block"
    assert guard.check_action("restart_service", [], no_alternate) is None


@pytest.mark.parametrize("engine", ENGINES)
def test_a_recovered_verdict_the_metrics_contradict_is_refused(engine, env, make_manager):
    """The Verifier is a model; it does not get to close an incident whose numbers are still bad."""
    from src.models.schemas import VerificationResult

    env(IC_ENGINE=engine)

    class AlwaysRecovered(FakeLLMClient):
        def structured(self, schema, *args, **kwargs):
            if schema is VerificationResult:
                self.calls.append((kwargs.get("agent", "unknown"), schema.__name__))
                return VerificationResult(
                    recovered=True,
                    confidence=0.99,
                    observations=["looks fine"],
                    expected_signal_met=True,
                    signal_assessment="claimed",
                    next_step="resolve",
                )
            return super().structured(schema, *args, **kwargs)

    # cascading_failure needs two actions; after the first the error rate is still high.
    state = asyncio.run(drive(make_manager(AlwaysRecovered()), "cascading_failure", "approve"))
    executed = [a for a in state.actions if (a.result or {}).get("applied")]
    assert len(executed) == 2, "the first, premature 'recovered' must not have ended the incident"
    assert any(f.get("kind") == "recovery_not_confirmed" for f in state.guardrail_flags)
    assert state.status == IncidentStatus.RESOLVED

"""Who decided? Every human-gate outcome names its decider, on both engines.

An agent that can roll back production must leave a record of WHO allowed each
action, not just that it was allowed. These tests pin that record down: a named
operator, policy auto-approval, a timeout, and a cancel each leave a distinct mark
on the action, the incident, and the postmortem.
"""

from __future__ import annotations

import asyncio

import pytest

from src.models.state import ActionStatus, EventType, IncidentStatus

from ._fake_llm import FakeLLMClient

ENGINES = ["loop", "graph"]


async def _run(manager, scenario_id, *, actor=None, decision="approve", auto_approve=False, cancel_by=None):
    state = manager.trigger(scenario_id, None, auto_approve=auto_approve, actor="starter")
    async for ev in manager.subscribe(state.incident_id):
        if ev.type == EventType.AWAITING_APPROVAL:
            if cancel_by:
                manager.request_cancel(state.incident_id, actor=cancel_by)
            elif decision:
                manager.resolve_approval(state.incident_id, ev.payload["action_id"], decision, "ok", actor=actor)
    return manager.get(state.incident_id)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_approver_is_named_on_the_action_and_in_the_postmortem(engine, env, make_manager):
    env(IC_ENGINE=engine)
    state = asyncio.run(_run(make_manager(), "bad_deploy", actor="alice"))
    assert state.created_by == "starter"
    action = state.actions[0]
    assert action.decided_by == "alice"
    assert action.decided_at is not None
    pm_action = state.postmortem["actions_taken"][0]
    assert pm_action["decided_by"] == "alice"
    assert state.postmortem["started_by"] == "starter"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_rejection_names_who_rejected(engine, env, make_manager):
    env(IC_ENGINE=engine)
    state = asyncio.run(_run(make_manager(), "bad_deploy", actor="bob", decision="reject"))
    assert state.actions[-1].status == ActionStatus.REJECTED
    assert state.actions[-1].decided_by == "bob"


@pytest.mark.parametrize("engine", ENGINES)
def test_policy_auto_approval_is_not_attributed_to_a_person(engine, env, make_manager):
    env(IC_ENGINE=engine)
    # A memory leak is fixed by a restart: medium risk, so auto-approve covers it.
    state = asyncio.run(_run(make_manager(), "memory_leak", decision=None, auto_approve=True))
    assert state.status == IncidentStatus.RESOLVED
    assert state.actions[0].decided_by == "policy:auto_approve"


def test_a_timeout_is_recorded_as_nobody(env):
    env(IC_APPROVAL_TIMEOUT_S="0.2")
    from src.loop import IncidentManager

    state = asyncio.run(_run(IncidentManager(llm=FakeLLMClient()), "bad_deploy", decision=None))
    assert state.actions[0].decided_by == "nobody:timeout"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_cancel_names_who_cancelled(engine, env, make_manager):
    env(IC_ENGINE=engine)
    state = asyncio.run(_run(make_manager(), "bad_deploy", cancel_by="carol"))
    assert state.status == IncidentStatus.CANCELLED
    assert state.cancelled_by == "carol"
    assert state.postmortem["cancelled_by"] == "carol"


@pytest.mark.parametrize("engine", ENGINES)
def test_spend_is_reported_before_the_run_waits_at_the_gate(engine, env, make_manager):
    """Found in a live console run: the cost meter read $0.0000 at the gate after
    triage, diagnosis, remediation and the Critic had all been paid for."""
    env(IC_ENGINE=engine, IC_APPROVAL_TIMEOUT_S="5")  # a regression fails fast, not hangs
    manager = make_manager()

    async def go():
        state = manager.trigger("bad_deploy", None, auto_approve=False, actor="starter")
        waiting = None
        async for ev in manager.subscribe(state.incident_id):
            if waiting is not None:
                assert ev.type == EventType.TOKEN_USAGE, f"expected usage at the gate, got {ev.type}"
                manager.resolve_approval(state.incident_id, waiting, "approve", "ok", actor="alice")
                waiting = None
            elif ev.type == EventType.AWAITING_APPROVAL:
                waiting = ev.payload["action_id"]
        return manager.get(state.incident_id)

    assert asyncio.run(go()).status == IncidentStatus.RESOLVED

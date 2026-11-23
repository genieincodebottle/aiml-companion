"""The integration seams: protocol conformance, and the action contracts.

The point of these is drift. During the migration described in ADR-0010 there are
two descriptions of the action menu - the `ActionType` enum and the registry - and
two descriptions of one concept always diverge. An action that exists in the enum
but has no registry entry is an action with no risk class, no preconditions and no
verification signal: ungoverned, while looking perfectly normal on the menu.

Catching that at CI rather than at 3am is the entire job here.
"""

from __future__ import annotations

import pytest

from src.integrations import ActionExecutor, ActionResult, TelemetrySource, allowed_for_role
from src.integrations.registry import DEFAULT_ACTIONS, ActionSpec, get_action, verify_matches_enum
from src.models.state import ActionRisk, ActionType
from src.sandbox.scenarios import get_scenario
from src.sandbox.stack import SandboxStack


@pytest.fixture
def stack():
    return SandboxStack(scenario=get_scenario("bad_deploy"), service="api-gateway")


# --- protocol conformance ---------------------------------------------------


def test_sandbox_satisfies_both_protocols(stack):
    """The reference implementation must actually conform, or the claim is empty.

    This caught a real gap: `SandboxStack` had no `supports()`, so the protocol it
    was supposed to exemplify did not accept it.
    """
    assert isinstance(stack, TelemetrySource)
    assert isinstance(stack, ActionExecutor)


def test_supports_rejects_an_unknown_action(stack):
    assert stack.supports("restart_service")
    assert not stack.supports("rm_minus_rf")


def test_metrics_never_leak_ground_truth(stack):
    """`faulted` is the answer key. A telemetry source that returns it makes every
    scenario trivial while still appearing to be solved by reasoning."""
    from src.tools.observability import observe_all

    assert "faulted" not in observe_all(stack)["metrics"]


# --- ActionResult -----------------------------------------------------------


def test_action_result_defaults_to_unknown_effectiveness():
    """`effective=None` means "too early to say", and it is the honest default.

    A real backend cannot answer this at the moment an action returns. Defaulting
    to True would make every adapter optimistic by accident, which is the single
    easiest way to make an autonomous remediator dangerous.
    """
    r = ActionResult(applied=True, reason="scaled to 5 replicas")
    assert r.effective is None
    assert r.to_dict()["effective"] is None


def test_action_result_can_express_partial_application():
    """Three of five pods came up: the world has moved but is in neither state."""
    r = ActionResult(applied=False, partial=True, reason="3/5 replicas ready", undo_token="rev-41")
    d = r.to_dict()
    assert d["partial"] is True and d["applied"] is False
    assert d["undo_token"] == "rev-41"


# --- registry ---------------------------------------------------------------


def test_registry_matches_the_action_enum():
    """The drift guard. See this module's docstring for why it matters."""
    verify_matches_enum()


def test_every_action_declares_a_risk_and_a_verification_signal():
    for name, spec in DEFAULT_ACTIONS.items():
        assert isinstance(spec.risk, ActionRisk), f"{name} has no risk class"
        if name != ActionType.NO_OP.value:
            assert spec.verification_signal, f"{name} declares no signal to verify against"


def test_high_risk_actions_are_restricted_to_the_full_role():
    """Fail-closed by construction: a restricted bot must not reach rollback or
    failover, and the registry is now where that is stated."""
    restricted = allowed_for_role("sre-bot-restricted")
    assert ActionType.ROLLBACK_DEPLOY.value not in restricted
    assert ActionType.FAILOVER.value not in restricted


def test_unknown_role_gets_nothing():
    """Same rule as the agent's RBAC: unknown means no permissions, never a
    permissive default."""
    assert allowed_for_role("captain") == frozenset()
    assert allowed_for_role("") == frozenset()


def test_read_only_role_can_only_do_nothing():
    assert allowed_for_role("read-only") == frozenset({ActionType.NO_OP.value})


def test_registry_roles_are_a_subset_of_the_guardrail_allowlists():
    """The registry must not grant something `ROLE_ALLOWED` forbids.

    While both exist, the guardrail layer is authoritative. A registry that is
    broader than it would be silently claiming permissions the code will refuse.
    """
    from src.guardrails.manager import ROLE_ALLOWED

    for role, allowed in ROLE_ALLOWED.items():
        from_registry = allowed_for_role(role)
        assert from_registry <= {a.value if hasattr(a, "value") else a for a in allowed}, (
            f"registry grants '{role}' actions the guardrail layer does not"
        )


# --- preconditions ----------------------------------------------------------


def test_failover_is_blocked_without_a_healthy_alternate():
    """This was previously only a sentence in a prompt, which a live run showed is
    not enough - the model proposed failover with no evidence an alternate existed,
    because nothing could stop it.

    The fixtures use the REAL shape `get_dependency_health` returns ({provider,
    status, ...}). An earlier version of this test invented a `healthy_alternate`
    key that no telemetry source produces - which "passed" while guaranteeing the
    precondition could never approve a failover against actual telemetry.
    """
    spec = get_action(ActionType.FAILOVER.value)
    assert spec.blocked_by({}) is not None
    degraded_only = [{"provider": "stripe-primary", "status": "degraded", "success_rate": 0.61}]
    assert spec.blocked_by({"dependency_health": degraded_only}) is not None
    with_alternate = degraded_only + [{"provider": "stripe-backup", "status": "healthy", "success_rate": 0.999}]
    assert spec.blocked_by({"dependency_health": with_alternate}) is None


def test_rollback_is_blocked_without_a_correlated_deploy():
    spec = get_action(ActionType.ROLLBACK_DEPLOY.value)
    assert spec.blocked_by({"deploys": []}) is not None
    assert spec.blocked_by({"deploys": [{"id": "d1", "within_incident_window": False}]}) is not None
    assert spec.blocked_by({"deploys": [{"id": "d1", "within_incident_window": True}]}) is None


def test_a_raising_precondition_refuses_rather_than_permits():
    """Fail-closed on a broken check. A predicate whose own bug silently grants
    permission is the worst outcome available here."""

    def boom(_obs):
        raise RuntimeError("telemetry backend unreachable")

    spec = ActionSpec(name="x", description="", risk=ActionRisk.HIGH, preconditions=(boom,))
    reason = spec.blocked_by({})
    assert reason is not None and "refusal" in reason


def test_irreversible_actions_are_marked_as_such():
    """You cannot un-restart a process or un-flush a cache. A design that assumes
    undo exists is worse than one that knows it usually does not."""
    assert get_action(ActionType.RESTART_SERVICE.value).reversible is False
    assert get_action(ActionType.CLEAR_CACHE.value).reversible is False
    assert get_action(ActionType.SCALE_OUT.value).reversible is True


def test_settle_time_is_nonzero_for_anything_that_takes_time():
    """Zero settle time is only honest for a simulation. An agent that reads a
    metric one second after a restart is reading noise."""
    for name in (
        ActionType.RESTART_SERVICE.value,
        ActionType.SCALE_OUT.value,
        ActionType.ROLLBACK_DEPLOY.value,
        ActionType.FAILOVER.value,
    ):
        assert get_action(name).settle_seconds > 0, f"{name} claims to verify instantly"

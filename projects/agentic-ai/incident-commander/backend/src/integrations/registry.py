"""The action registry: what an action IS, declared as data.

Today the menu is an enum in `models/state.py`, with its risk in
`guardrails/manager.py` and its permissions in `ROLE_ALLOWED`. Three places, one
concept. That works for six fixed actions and stops working the moment an
integrator needs their own runbook, because the pieces live in files they should
not be editing.

Here an action declares its own contract in one place, and the pieces the rest of
the system needs are *derived* from it. Crucially this keeps ADR-0003's property:
the model still chooses from a closed menu and out-of-menu output is still a
decoding error. The menu is simply built from data instead of hardcoded.

Two fields carry the weight, and neither exists in the current design:

* **`preconditions`** turns "do not fail over without a healthy alternate" from a
  sentence in a prompt - which a model may ignore, and which a prompt-injected
  model certainly will - into a callable the loop checks before offering the
  action.
* **`settle_seconds`** is the admission that verification takes time. The sandbox
  answers "did it work?" instantly because it owns the truth. A real backend does
  not, and an agent that checks a metric one second after a restart will read noise
  as recovery.

Status: the PRECONDITIONS are enforced today - `GuardrailManager.check_action`
calls `get_action(name).blocked_by(facts)` on every proposal, in both engines, with
facts the loop fetches itself (`tools.observability.precondition_facts`). The menu
itself still comes from the `ActionType` enum; making this registry its source is
the migration described in ADR-0010, and `verify_matches_enum()` guards the two
against drift until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..models.state import ActionRisk, ActionType

#: A precondition inspects current telemetry and returns None if the action is
#: allowed, or a human-readable reason why not. Returning a reason rather than a
#: bool matters: that string goes to the operator and into the postmortem.
Precondition = Callable[[dict[str, Any]], str | None]


@dataclass(frozen=True)
class ActionSpec:
    """Everything the system needs to know about one remediation action."""

    name: str
    description: str

    #: Assigned here, never by the model. See ADR-0002.
    risk: ActionRisk

    #: True only when there is a real undo path AND the executor implements
    #: `ReversibleExecutor`. Defaults to False because most actions are not
    #: reversible, and a wrong default here is a wrong default in the direction
    #: that gets someone paged.
    reversible: bool = False

    #: What the Verifier should look for if this action worked. Declared by the
    #: action rather than invented per incident, so a model cannot quietly pick an
    #: easier-to-satisfy prediction than the one the action deserves.
    verification_signal: str = ""

    #: How long to wait before that signal means anything. Zero is only honest for
    #: a simulation; a restart needs a warm-up, a scale-out needs scheduling, a
    #: cache clear needs refill.
    settle_seconds: int = 0

    #: What else this can affect, in plain words. Shown at the human gate, where
    #: "restarts every pod in the deployment" is the sentence that changes a
    #: decision.
    blast_radius: str = ""

    #: Checked before the action is offered. Each returns None (allowed) or a
    #: reason string.
    preconditions: tuple[Precondition, ...] = ()

    #: Roles permitted to run this. Feeds the existing RBAC allowlist.
    allowed_roles: frozenset[str] = frozenset({"sre-bot"})

    #: Extra context for an adapter (namespace, cluster, deployment selector).
    adapter_hints: dict[str, Any] = field(default_factory=dict)

    def blocked_by(self, observations: dict[str, Any]) -> str | None:
        """First precondition that refuses, or None.

        Fail-closed on a broken precondition: a check that raises is treated as a
        refusal, not as a pass. A predicate whose own bug silently grants
        permission is the worst outcome available here.
        """
        for check in self.preconditions:
            try:
                reason = check(observations)
            except Exception as exc:  # noqa: BLE001 - a broken check must not permit
                return f"precondition check failed and is treated as a refusal: {exc}"
            if reason:
                return reason
        return None


# --- preconditions used by the default catalog ------------------------------


def _requires_healthy_alternate(obs: dict[str, Any]) -> str | None:
    """Failover needs somewhere healthy to fail over TO.

    This was previously only a sentence in the Remediation prompt. A live run
    showed why that is not enough: the model proposed failover with no evidence an
    alternate existed, because nothing could stop it.
    """
    deps = [d for d in (obs.get("dependency_health") or []) if isinstance(d, dict)]
    if not deps:
        return "no dependency-health data, so a healthy alternate cannot be confirmed"
    # An alternate is a provider entry reporting healthy. This must match the shape
    # `get_dependency_health` actually returns ({provider, status, ...}) - an
    # earlier version looked for a `healthy_alternate` key that no telemetry source
    # produces, which would have refused failover on the one scenario whose correct
    # action IS failover, the moment this registry became authoritative.
    if not any(d.get("status") == "healthy" for d in deps):
        return "no healthy alternate is available to fail over to"
    return None


def _requires_recent_deploy(obs: dict[str, Any]) -> str | None:
    """Rolling back is only meaningful if something was deployed recently."""
    deploys = obs.get("deploys") or []
    if not any(d.get("within_incident_window") for d in deploys if isinstance(d, dict)):
        return "no deploy correlates with the incident window, so there is nothing to roll back to"
    return None


# --- the default catalog ----------------------------------------------------
#
# Mirrors the current hardcoded menu, now with the contract each action always
# had implicitly. An integrator replaces this wholesale.

DEFAULT_ACTIONS: dict[str, ActionSpec] = {
    ActionType.NO_OP.value: ActionSpec(
        name=ActionType.NO_OP.value,
        description="Take no action. The correct answer when the service is healthy.",
        risk=ActionRisk.LOW,
        reversible=True,
        verification_signal="metrics remain within normal bounds",
        blast_radius="none",
        allowed_roles=frozenset({"sre-bot", "sre-bot-restricted", "read-only"}),
    ),
    ActionType.CLEAR_CACHE.value: ActionSpec(
        name=ActionType.CLEAR_CACHE.value,
        description="Flush the service cache to recover from poisoning or a stampede.",
        risk=ActionRisk.LOW,
        reversible=False,  # a flushed cache cannot be un-flushed
        verification_signal="cache hit ratio recovers and origin load falls",
        settle_seconds=60,  # the cache has to refill before the ratio means anything
        blast_radius="a short latency spike for every client while the cache refills",
        allowed_roles=frozenset({"sre-bot", "sre-bot-restricted"}),
    ),
    ActionType.RESTART_SERVICE.value: ActionSpec(
        name=ActionType.RESTART_SERVICE.value,
        description="Roll the service processes. Clears leaked memory and wedged state.",
        risk=ActionRisk.MEDIUM,
        reversible=False,  # you cannot un-restart a process
        verification_signal="memory returns to baseline and error rate falls",
        settle_seconds=90,
        blast_radius="in-flight requests on restarted instances are dropped",
        allowed_roles=frozenset({"sre-bot", "sre-bot-restricted"}),
    ),
    ActionType.SCALE_OUT.value: ActionSpec(
        name=ActionType.SCALE_OUT.value,
        description="Add capacity to relieve saturation.",
        risk=ActionRisk.MEDIUM,
        reversible=True,  # scale back in
        verification_signal="saturation and queue depth fall without error-rate regression",
        settle_seconds=120,  # new instances must schedule, start and pass health checks
        blast_radius="increased cost; more connections against shared downstreams",
        allowed_roles=frozenset({"sre-bot", "sre-bot-restricted"}),
    ),
    ActionType.ROLLBACK_DEPLOY.value: ActionSpec(
        name=ActionType.ROLLBACK_DEPLOY.value,
        description="Return the service to the previous known-good release.",
        risk=ActionRisk.HIGH,
        reversible=True,  # roll forward again
        verification_signal="the error signature introduced by the deploy disappears",
        settle_seconds=180,
        blast_radius="every instance; any data written by the new version may be incompatible",
        preconditions=(_requires_recent_deploy,),
        allowed_roles=frozenset({"sre-bot"}),
    ),
    ActionType.FAILOVER.value: ActionSpec(
        name=ActionType.FAILOVER.value,
        description="Shift traffic to a healthy alternate region or provider.",
        risk=ActionRisk.HIGH,
        reversible=True,  # fail back
        verification_signal="error rate falls as traffic drains to the alternate",
        settle_seconds=180,
        blast_radius="whole-region traffic shift; replication lag may surface as stale reads",
        preconditions=(_requires_healthy_alternate,),
        allowed_roles=frozenset({"sre-bot"}),
    ),
}


def get_action(name: str) -> ActionSpec | None:
    return DEFAULT_ACTIONS.get(name)


def allowed_for_role(role: str) -> frozenset[str]:
    """The action names a role may run. Fail-closed on an unknown role."""
    return frozenset(n for n, spec in DEFAULT_ACTIONS.items() if role in spec.allowed_roles)


def verify_matches_enum() -> None:
    """Assert the registry and `ActionType` describe the same menu.

    During the migration both exist, and two descriptions of one concept drift.
    A test calls this so the drift is caught at CI rather than at 3am, when an
    action in the enum has no contract and is therefore ungoverned.
    """
    from_enum = {a.value for a in ActionType}
    from_registry = set(DEFAULT_ACTIONS)
    if from_enum != from_registry:
        missing = from_enum - from_registry
        extra = from_registry - from_enum
        raise AssertionError(
            f"action registry has drifted from ActionType: missing {sorted(missing)}, extra {sorted(extra)}"
        )

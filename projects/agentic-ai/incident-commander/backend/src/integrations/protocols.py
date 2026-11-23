"""The two seams between this agent loop and a real production estate.

Everything the agents perceive comes through `TelemetrySource`. Everything that
changes the world goes through `ActionExecutor`. Nothing else in the system needs
to know whether it is driving a simulation or a cluster.

These are `Protocol`s rather than base classes on purpose: an integrator writes a
plain object with the right methods and never imports from this package. That also
means `SandboxStack` already satisfies both without inheriting anything - it is the
reference implementation, not a special case.

See ADR-0010 for why this is shaped the way it is, and `docs/integration.md` for a
worked adapter.

## What these protocols deliberately do NOT solve

**Verification.** `ActionResult.effective` is `bool | None`, and `None` - "cannot
yet tell" - is the honest answer for most real backends at the moment an action
returns. The sandbox can answer instantly because it owns the ground truth; a real
cluster cannot. Deciding whether a metric moved because of your action or because
traffic dropped is a statistics problem this project does not solve, and pretending
otherwise by returning a confident `True` is the single easiest way to make an
autonomous remediator dangerous.

**Partial application.** `ActionResult.applied` can be `False` while the world has
still changed - a scale-out that brought up three of five pods. The field exists so
an adapter can say so; reconciling that state is backend-specific work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionResult:
    """What happened when an action was executed.

    Deliberately separates four things that are easy to conflate:

    * `applied`   - did the change reach the system at all?
    * `effective` - did it fix the incident? `None` means "too early to say",
                    which is the correct answer for most real backends.
    * `progress`  - it helped without finishing. Compound faults need this.
    * `harmful`   - it made things worse. Tracked because the evaluation's
                    harmful-action rate is the metric that actually matters.
    """

    applied: bool
    reason: str = ""
    effective: bool | None = None
    progress: bool = False
    harmful: bool = False
    #: True when the change landed only in part. The world has moved but is in
    #: neither the before nor the after state, and something must reconcile it.
    partial: bool = False
    #: Backend-specific handle for undo, when the action declared itself
    #: reversible. A deployment revision, a snapshot id, a previous replica count.
    undo_token: str | None = None
    #: Anything the adapter wants to carry back for the audit trail.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The shape `loop.py` and the event stream already expect."""
        return {
            "applied": self.applied,
            "effective": self.effective,
            "progress": self.progress,
            "harmful": self.harmful,
            "partial": self.partial,
            "reason": self.reason,
            **({"undo_token": self.undo_token} if self.undo_token else {}),
            **self.detail,
        }


@runtime_checkable
class TelemetrySource(Protocol):
    """Read-only perception. Implementations MUST be side-effect free.

    That is not a style note. The read/write split is what lets diagnosis run
    unguarded while every action passes a gate: if reading could change
    production, the guardrail layer would be protecting nothing.

    Method names match `SandboxStack`, so it already conforms. Only `metrics`,
    `logs` and `traces` are required; the rest are optional and a source that
    lacks one should raise `NotImplementedError` - `run_tool` turns that into an
    error dict rather than killing the loop, which is the correct degradation when
    a telemetry backend is unavailable mid-incident.
    """

    def metrics(self) -> dict[str, Any]:
        """Current metric snapshot: latency, error rate, saturation.

        MUST NOT include a ground-truth health flag. Agents are supposed to infer
        health from the numbers, exactly as an on-call engineer does; leaking a
        boolean answer makes every scenario trivial while still looking solved.
        """
        ...

    def logs(self, limit: int = 20) -> list[str]:
        """Recent log lines. UNTRUSTED: attacker- and user-influenced text.

        Whatever this returns is sanitized by the guardrail layer before it
        reaches a prompt. An adapter must not pre-trust it.
        """
        ...

    def traces(self) -> dict[str, Any]:
        """Trace summary: which span is slow or erroring. Also untrusted text."""
        ...


@runtime_checkable
class ActionExecutor(Protocol):
    """The write side. The only thing in the system that changes the world.

    An implementation is called by exactly one function
    (`tools.remediation.execute_remediation`) and only after the guardrail layer
    has cleared the action. It should still validate independently: defence in
    depth costs little here, and an executor that trusts its caller is one
    refactor away from being called from somewhere new.
    """

    def apply_action(self, action_type: str, **kwargs: Any) -> dict[str, Any]:
        """Execute an approved action. MUST NOT raise on a rejected action.

        Return an `ActionResult.to_dict()`. Raising for an ordinary refusal turns
        "I would not do that" into a crash; the loop is built to read a refusal and
        re-diagnose.

        MUST be idempotent per `action_id` where the backend allows it. A retried
        approval must not scale out twice.
        """
        ...

    def supports(self, action_type: str) -> bool:
        """Whether this executor can run that action at all.

        Checked before the action is offered, so an estate without a failover path
        never sees `failover` on the menu rather than proposing it and failing.
        """
        ...


@runtime_checkable
class ReversibleExecutor(ActionExecutor, Protocol):
    """An executor that can undo what it did.

    Separate from `ActionExecutor` because most real actions are NOT reversible -
    you cannot un-restart a process - and a design that assumes undo exists is
    worse than one that knows it usually does not. The action registry's
    `reversible` flag should be true only when the executor implements this AND the
    specific action genuinely has an undo path.
    """

    def undo(self, action_type: str, undo_token: str) -> dict[str, Any]:
        """Revert a previously applied action using its `undo_token`."""
        ...

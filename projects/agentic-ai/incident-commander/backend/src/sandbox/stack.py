"""The simulated microservice stack.

Holds per-service health, a deploy history, config state, dependency health, and a
log buffer. While a fault is active, telemetry reflects the scenario's signature.
Remediation actions mutate the stack: the *correct* action for the active fault
advances the remediation plan, and completing the plan clears the fault. Any other
action is recorded as executed-but-ineffective, which is exactly what drives the
agent loop to iterate.

Two properties are load-bearing for the teaching goal:

1. **Ground truth is hidden.** `metrics()` never exposes the internal `faulted`
   flag (see `tools.observability.observe_all`). Agents must infer health from
   error rate and latency, like a real on-call engineer.
2. **Some signals cost a tool call.** Pool stats, config diff, dependency health
   and SLO burn are NOT in the default telemetry bundle. An agent that needs them
   has to ask for them, which is what makes the evidence-gathering loop real
   rather than decorative.

This is a sandbox: actions never touch anything real. That is the whole point of
the human-approval gate in the real product, modelled here safely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .scenarios import Scenario

# Baseline healthy metrics for any service.
_HEALTHY = {
    "p99_latency_ms": 180,
    "error_rate": 0.001,
    "cpu": 0.32,
    "mem": 0.41,
    "db_pool_in_use": 6,
    "db_pool_max": 50,
    "requests_per_sec": 420,
}

# The action menu the agent may choose from (mirrors models.state.ActionType).
VALID_ACTIONS = {
    "rollback_deploy",
    "restart_service",
    "scale_out",
    "clear_cache",
    "failover",
    "no_op",
}

#: SLO the burn-rate tool is measured against.
_SLO_TARGET_AVAILABILITY = 0.999


@dataclass
class SandboxStack:
    """One isolated simulated stack per incident."""

    scenario: Scenario
    service: str
    created_at: float = field(default_factory=time.time)
    faulted: bool = True
    plan_index: int = 0  # how many ordered remediation steps have completed
    deploys: list[dict[str, Any]] = field(default_factory=list)
    #: Every tool call made against this stack, for the audit trail and the eval.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    _logs: list[str] = field(default_factory=list)
    _recovered_at: float | None = None
    _restarts: int = 0
    _replicas: int = 3

    def __post_init__(self) -> None:
        # `false_alarm` is genuinely healthy: the page was spurious.
        if self.scenario.starts_healthy:
            self.faulted = False
        self.deploys = [dict(d) for d in (self.scenario.deploys or _DEFAULT_DEPLOY_HISTORY)]
        self._logs = [f"[INFO] {self.service} healthy at startup"]
        if self.scenario.starts_healthy:
            self._logs.append(f"[WARN] {self.scenario.log_signature}")
            self._logs.append(f"[INFO] {self.service} all real signals nominal")
        else:
            self._logs.append(f"[WARN] {self.scenario.log_signature}")
            self._logs.append(f"[ERROR] {self.scenario.log_signature}")
        # Attacker-controlled lines, interleaved so they look like ordinary log noise.
        # The guardrail layer is responsible for neutralizing these.
        for line in self.scenario.injected_logs:
            self._logs.append(line)

    # --- phase-aware telemetry (telemetry evolves as plan steps complete) ----

    def _phase_signature(self) -> str:
        if self.plan_index == 0:
            return self.scenario.log_signature
        idx = self.plan_index - 1
        pt = self.scenario.phase_telemetry
        if idx < len(pt):
            return pt[idx].get("log_signature", self.scenario.log_signature)
        return self.scenario.log_signature

    def _phase_metrics(self) -> dict[str, Any]:
        if self.plan_index == 0:
            return self.scenario.metric_overrides
        idx = self.plan_index - 1
        pt = self.scenario.phase_telemetry
        if idx < len(pt):
            return pt[idx].get("metric_overrides", self.scenario.metric_overrides)
        return self.scenario.metric_overrides

    # --- read side: telemetry the observability tools expose ----------------

    def _note_tool(self, name: str) -> None:
        self.tool_calls.append({"tool": name, "ts": time.time(), "plan_index": self.plan_index})

    def _metrics_raw(self) -> dict[str, Any]:
        """Metric snapshot without recording a tool call (used internally)."""
        m = dict(_HEALTHY)
        if self.faulted:
            m.update(self._phase_metrics())
        m["replicas"] = self._replicas
        return {"service": self.service, "faulted": self.faulted, **m}

    def _metrics_visible(self) -> dict[str, Any]:
        """`_metrics_raw` minus the ground-truth `faulted` flag.

        Every snapshot that can reach an agent goes through here: the metrics tool,
        and the `metrics_after` attached to action results (which lands in
        `state.actions`, and from there in prompts and the API). Filtering only in
        `observe_all` was not enough - the Diagnostician can request `get_metrics`
        on demand, and that path handed back the answer key.
        """
        return {k: v for k, v in self._metrics_raw().items() if k != "faulted"}

    def metrics(self) -> dict[str, Any]:
        self._note_tool("get_metrics")
        return self._metrics_visible()

    def logs(self, limit: int = 20) -> list[str]:
        self._note_tool("get_logs")
        if not self.faulted and self._recovered_at is not None:
            tail = [f"[INFO] {self.service} recovered; metrics nominal"]
            return (self._logs + tail)[-limit:]
        return self._logs[-limit:]

    def traces(self) -> dict[str, Any]:
        self._note_tool("get_traces")
        if self.faulted:
            return {
                "service": self.service,
                "slow_span": self.scenario.default_service,
                "error_span_rate": self._phase_metrics().get("error_rate", 0.0),
                "note": self._phase_signature(),
            }
        return {"service": self.service, "slow_span": None, "error_span_rate": 0.0, "note": "nominal"}

    #: A deploy shipped within this many minutes of the incident is treated as
    #: correlated. Anything older is explicitly labelled as predating it.
    RECENT_DEPLOY_WINDOW_MIN = 45

    def deploy_history(self) -> list[dict[str, Any]]:
        """Recent deploys, each annotated with whether it correlates with the incident.

        The annotation is the point. Handed a bare `minutes_ago` number, a model has
        to do arithmetic against an incident-start time it was never given, and it
        reliably gets this wrong: a live run called a 190-minute-old release "the
        recent deployment" and proposed a rollback for a connection-pool fault.

        A real deploy-tracking API would tell you what shipped inside the incident
        window rather than making the caller infer it, so this one does too. The
        general lesson is worth more than the fix: never make the model do implicit
        arithmetic against a reference point you have not put in front of it.
        """
        self._note_tool("get_deploy_history")
        annotated = []
        for deploy in self.deploys:
            age = deploy.get("minutes_ago", 0)
            correlated = age <= self.RECENT_DEPLOY_WINDOW_MIN
            annotated.append(
                {
                    **deploy,
                    "within_incident_window": correlated,
                    "recency": (
                        f"shipped {age} min ago - INSIDE the incident window, so it may be the cause"
                        if correlated
                        else f"shipped {age} min ago ({age / 60:.1f}h) - PREDATES the incident; "
                        "the service was healthy after this deploy, so it is unlikely to be the cause"
                    ),
                }
            )
        return annotated

    # --- on-demand tools: NOT in the default telemetry bundle ---------------

    def pool_stats(self) -> dict[str, Any]:
        """Connection-pool detail. Distinguishes "pool too small" from "pool leaking"."""
        self._note_tool("get_pool_stats")
        m = self._metrics_raw()
        in_use, maximum = m.get("db_pool_in_use", 0), m.get("db_pool_max", 50)
        waiters = max(0, in_use - maximum + 12) if self.faulted and in_use >= maximum else 0
        leaked = bool(self.faulted and self.plan_index > 0 and "leak" in self._phase_signature())
        # Two numbers, because one cannot tell the cases apart. WAIT is how long a request
        # queues for a free connection; it is bounded by the latency the service actually
        # shows. HOLD is how long a connection stays checked out: normal when the pool is
        # just too small, very long when connections leak. An earlier single
        # "mean_checkout_ms: 30000" contradicted a 3.4 s p99 and read as slow queries, so a
        # live Critic correctly vetoed scale_out on a fixture that argued against itself.
        return {
            "service": self.service,
            "in_use": in_use,
            "max": maximum,
            "utilization": round(in_use / maximum, 3) if maximum else None,
            "waiting_threads": waiters,
            "mean_wait_ms": round(m.get("p99_latency_ms", 0) * 0.8) if waiters else 4,
            "mean_hold_ms": 30000 if leaked else 40,
            "leaked_suspected": leaked,
        }

    def config_diff(self) -> list[dict[str, Any]]:
        """Recent configuration changes. Some faults are visible ONLY here."""
        self._note_tool("get_config_diff")
        if not self.scenario.config_diff:
            return [{"note": "no configuration changes in the last 24h"}]
        return [dict(c) for c in self.scenario.config_diff]

    def dependency_health(self) -> list[dict[str, Any]]:
        """Health of downstream providers, with a healthy alternate when one exists."""
        self._note_tool("get_dependency_health")
        if self.scenario.dependency_health and self.faulted:
            return [dict(d) for d in self.scenario.dependency_health]
        return [{"provider": "all", "status": "healthy", "p99_ms": 210, "success_rate": 0.999}]

    def slo_burn(self) -> dict[str, Any]:
        """Error-budget burn rate - the signal that tells you how urgent this is."""
        self._note_tool("get_slo_burn")
        m = self._metrics_raw()
        error_rate = float(m.get("error_rate", 0.0))
        budget = 1.0 - _SLO_TARGET_AVAILABILITY
        burn = round(error_rate / budget, 2) if budget else 0.0
        return {
            "service": self.service,
            "slo_target_availability": _SLO_TARGET_AVAILABILITY,
            "current_error_rate": error_rate,
            "burn_rate_x": burn,
            "budget_exhausted_in_hours": round(720 / burn, 2) if burn > 0 else None,
            "note": "burn_rate_x above 14.4 means the 30-day budget is gone in under 2 days",
        }

    def is_healthy(self) -> bool:
        return not self.faulted

    # --- write side: remediation actions ------------------------------------

    def supports(self, action_type: str) -> bool:
        """Whether this stack can run that action at all.

        Part of the `integrations.ActionExecutor` protocol. The sandbox supports
        the whole menu; a real executor would return False for actions its estate
        has no path for, so they are never offered rather than proposed and failed.
        """
        return action_type in VALID_ACTIONS

    def apply_action(self, action_type: str, **_kwargs: Any) -> dict[str, Any]:
        """Apply a remediation action against the ordered remediation plan.

        The plan must be completed IN ORDER. Applying the right next step advances
        the plan (and reveals new telemetry); the final step clears the fault. Any
        other action is recorded as executed-but-ineffective, which the Verifier
        catches and the loop then re-diagnoses.
        """
        if action_type not in VALID_ACTIONS:
            return {
                "applied": False,
                "effective": False,
                "progress": False,
                "reason": f"unknown action '{action_type}'",
            }

        if action_type == "no_op":
            # On a healthy service, doing nothing IS the correct outcome.
            if not self.faulted:
                return {
                    "applied": True,
                    "effective": True,
                    "progress": True,
                    "reason": "no action taken; the service is healthy and needed no remediation",
                    "metrics_after": self._metrics_visible(),
                }
            return {"applied": True, "effective": False, "progress": False, "reason": "no-op took no action"}

        if not self.faulted:
            # Acting on a healthy service is a mistake worth naming loudly: in
            # production this is how an agent turns a false alarm into an outage.
            self._logs.append(f"[WARN] action '{action_type}' applied to a HEALTHY service; unnecessary disruption")
            return {
                "applied": True,
                "effective": False,
                "progress": False,
                "harmful": True,
                "reason": (
                    # Says what happened, not what the right answer was: this text
                    # reaches the agents' prompts, and an answer key there would make
                    # the next proposal trivially correct.
                    f"'{action_type}' was applied to a healthy service - unnecessary and disruptive"
                ),
                "metrics_after": self._metrics_visible(),
            }

        self._record_side_effects(action_type)

        if self.scenario.unfixable:
            self._logs.append(
                f"[WARN] action '{action_type}' applied but the fault is not in our control; nothing changed"
            )
            return {
                "applied": True,
                "effective": False,
                "progress": False,
                "reason": (
                    f"'{action_type}' completed but the underlying fault is external and cannot be "
                    "remediated from here; human escalation is required"
                ),
                "metrics_after": self._metrics_visible(),
            }

        plan = self.scenario.plan
        expected = plan[self.plan_index]

        if action_type == expected:
            self.plan_index += 1
            step_no, total = self.plan_index, len(plan)
            if self.plan_index >= total:
                self.faulted = False
                self._recovered_at = time.time()
                self._logs.append(
                    f"[INFO] action '{action_type}' applied (step {step_no}/{total}); fault cleared on {self.service}"
                )
                return {
                    "applied": True,
                    "effective": True,
                    "progress": True,
                    "reason": (
                        f"'{action_type}' completed the remediation plan ({step_no}/{total}); service recovered"
                    ),
                    "metrics_after": self._metrics_visible(),
                }
            # Advanced one step but not done: surface the next phase of telemetry.
            self._advance_phase_logs()
            return {
                "applied": True,
                "effective": False,
                "progress": True,
                "reason": (
                    f"'{action_type}' applied (step {step_no}/{total}); fault not yet cleared, further action required"
                ),
                "metrics_after": self._metrics_visible(),
            }

        self._logs.append(f"[WARN] action '{action_type}' applied but had no effect; fault persists")
        return {
            "applied": True,
            "effective": False,
            "progress": False,
            "reason": f"'{action_type}' did not address the active fault",
            "metrics_after": self._metrics_visible(),
        }

    def _record_side_effects(self, action_type: str) -> None:
        """Actions change the world even when they do not fix the fault."""
        if action_type == "restart_service":
            self._restarts += 1
        elif action_type == "scale_out":
            self._replicas += 2
        elif action_type == "rollback_deploy" and self.deploys:
            rolled = self.deploys[0].get("version")
            self._logs.append(f"[INFO] rolled back {rolled}; now serving {self.deploys[-1].get('version')}")

    def _advance_phase_logs(self) -> None:
        sig = self._phase_signature()
        self._logs.append(f"[WARN] {sig}")
        self._logs.append(f"[ERROR] {sig}")


# Default deploy history: a QUIET one, with nothing shipped recently.
#
# This default matters more than it looks. An earlier version put a 14-minute-old
# deploy in every scenario, including ones whose whole premise is "latency moved
# and nothing was deployed". A live model reads a fresh deploy as strong evidence
# and reasonably proposes a rollback - so those scenarios were unsolvable as
# written, and the fault was in the fixture, not the agent.
#
# A recent deploy is a signal. Scenarios that mean to give the agent that signal
# declare it explicitly via `Scenario.deploys`.
_DEFAULT_DEPLOY_HISTORY: tuple[dict[str, Any], ...] = (
    {"id": "deploy_a90", "version": "v2.8.0", "minutes_ago": 190, "author": "ci-bot"},
    {"id": "deploy_a89", "version": "v2.7.9", "minutes_ago": 1450, "author": "ci-bot"},
)

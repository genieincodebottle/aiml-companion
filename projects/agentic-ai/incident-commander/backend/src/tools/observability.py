"""Read-only observability tools - the agents' senses.

These are the only way agents perceive the system. They are intentionally
side-effect free: reading telemetry can never change production. That read/write
split is a core safety property of the design (diagnosis is free, action is gated).

The important structure here is `TOOL_REGISTRY`. It maps a `TelemetryTool` enum
member to a callable, a human-readable description, and a cost hint. Three things
fall out of that:

1. **The Diagnostician can request tools by name.** Its schema is bound to the
   same enum, so a requested tool always exists (see `run_tools`).
2. **`describe_tools()` renders the catalog into the prompt**, so the agent knows
   what it can ask for without that list being duplicated in prose.
3. **The default bundle is deliberately incomplete.** `observe_all` returns only
   metrics, logs, traces and deploys - the cheap, always-available signals. Pool
   stats, config diff, dependency health and SLO burn cost a round trip, exactly
   like a real observability stack where every query has a price. An agent that
   wants them must decide it needs them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, NamedTuple

from ..models.state import TelemetryTool
from ..sandbox.stack import SandboxStack

logger = logging.getLogger(__name__)


# --- individual tools -------------------------------------------------------


def get_metrics(stack: SandboxStack) -> dict[str, Any]:
    """Current metric snapshot for the implicated service."""
    return stack.metrics()


def get_logs(stack: SandboxStack, limit: int = 20) -> list[str]:
    """Recent log lines (tail). UNTRUSTED: contains user- and attacker-influenced text."""
    return stack.logs(limit=limit)


def get_traces(stack: SandboxStack) -> dict[str, Any]:
    """Distributed-trace summary highlighting the slow / error span."""
    return stack.traces()


def get_deploy_history(stack: SandboxStack) -> list[dict[str, Any]]:
    """Recent deploys, newest first. Key signal for regression-class incidents."""
    return stack.deploy_history()


def get_pool_stats(stack: SandboxStack) -> dict[str, Any]:
    """Connection-pool detail: utilization, waiters, checkout time, leak suspicion."""
    return stack.pool_stats()


def get_config_diff(stack: SandboxStack) -> list[dict[str, Any]]:
    """Configuration changes in the last 24h. Catches drift with no deploy."""
    return stack.config_diff()


def get_dependency_health(stack: SandboxStack) -> list[dict[str, Any]]:
    """Downstream provider health, including whether a healthy alternate exists."""
    return stack.dependency_health()


def get_slo_burn(stack: SandboxStack) -> dict[str, Any]:
    """Error-budget burn rate: how urgent this incident actually is."""
    return stack.slo_burn()


# --- registry ---------------------------------------------------------------


class ToolSpec(NamedTuple):
    fn: Callable[[SandboxStack], Any]
    description: str
    #: Rough relative expense. Surfaced to the agent so "ask for everything" is
    #: visibly not free - the same pressure a real on-call engineer feels.
    cost_hint: str
    #: True when the output contains text influenced by users or attackers and must
    #: therefore be sanitized before it reaches the model.
    untrusted: bool = False


TOOL_REGISTRY: dict[TelemetryTool, ToolSpec] = {
    TelemetryTool.METRICS: ToolSpec(get_metrics, "Current metrics: latency, error rate, CPU, memory, pool.", "cheap"),
    TelemetryTool.LOGS: ToolSpec(
        get_logs, "Tail of recent application logs. Free-text, untrusted.", "cheap", untrusted=True
    ),
    TelemetryTool.TRACES: ToolSpec(
        get_traces, "Trace summary: which span is slow or erroring.", "cheap", untrusted=True
    ),
    TelemetryTool.DEPLOY_HISTORY: ToolSpec(
        get_deploy_history, "Recent deploys with age and author. Use for regression-class faults.", "cheap"
    ),
    TelemetryTool.POOL_STATS: ToolSpec(
        get_pool_stats,
        "Connection-pool internals: waiting threads, checkout latency, leak suspicion. "
        "Use to tell 'pool too small' apart from 'pool leaking'.",
        "moderate",
    ),
    TelemetryTool.CONFIG_DIFF: ToolSpec(
        get_config_diff,
        "Configuration changes in the last 24h. Use when metrics moved but nothing deployed.",
        "moderate",
    ),
    TelemetryTool.DEPENDENCY_HEALTH: ToolSpec(
        get_dependency_health,
        # The trigger is deliberately phrased for the DIAGNOSTICIAN, which is the
        # only agent that can request a tool. An earlier wording said "use before
        # proposing failover", which addresses the Remediation agent - and created a
        # deadlock in a live run: Remediation may not propose failover without
        # evidence of a healthy alternate, but it cannot ask for that evidence, and
        # the Diagnostician had no reason to fetch it. The incident escalated with
        # the right answer unreachable.
        "Downstream provider health, and whether a healthy alternate exists. Request this whenever "
        "the telemetry implicates an external provider, upstream dependency, or third-party API - "
        "it is the only way to establish whether shifting traffic is even an option.",
        "moderate",
    ),
    TelemetryTool.SLO_BURN: ToolSpec(
        get_slo_burn, "Error-budget burn rate. Use to justify urgency and severity.", "cheap"
    ),
}

#: The always-on bundle. Cheap, and enough to diagnose most faults - but not all.
DEFAULT_BUNDLE: tuple[TelemetryTool, ...] = (
    TelemetryTool.METRICS,
    TelemetryTool.LOGS,
    TelemetryTool.TRACES,
    TelemetryTool.DEPLOY_HISTORY,
)

#: Tools whose output is untrusted text and must pass the guardrail sanitizer.
UNTRUSTED_TOOLS: frozenset[TelemetryTool] = frozenset(t for t, spec in TOOL_REGISTRY.items() if spec.untrusted)


def describe_tools(exclude: tuple[TelemetryTool, ...] = DEFAULT_BUNDLE) -> str:
    """Render the tool catalog for a prompt.

    By default the always-on bundle is excluded, because there is no point offering
    an agent a tool whose output it has already been handed.
    """
    lines = []
    for tool, spec in TOOL_REGISTRY.items():
        if tool in exclude:
            continue
        lines.append(f"- {tool.value} ({spec.cost_hint}): {spec.description}")
    return "\n".join(lines)


def run_tool(stack: SandboxStack, tool: TelemetryTool) -> Any:
    """Execute one read-only tool. Never raises: a broken tool returns an error dict.

    Returning an error rather than raising is deliberate. A telemetry backend being
    down is normal during an incident; it should degrade the agent's evidence, not
    crash the loop.
    """
    spec = TOOL_REGISTRY.get(tool)
    if spec is None:  # pragma: no cover - unreachable while the schema is bound to the enum
        return {"error": f"unknown tool '{tool}'"}
    try:
        return spec.fn(stack)
    except Exception as exc:  # noqa: BLE001 - a failing tool must not kill the loop
        logger.warning("tool %s failed: %s", tool.value, exc)
        return {"error": f"tool '{tool.value}' failed: {exc}"}


def run_tools(stack: SandboxStack, tools: list[TelemetryTool], *, limit: int = 4) -> dict[str, Any]:
    """Execute a batch of requested tools, capped at `limit`.

    The cap is a guardrail, not an optimization: without it an agent that reports
    low confidence can request every tool every round and turn a bounded loop into
    an unbounded one.
    """
    results: dict[str, Any] = {}
    for tool in tools[:limit]:
        results[tool.value] = run_tool(stack, tool)
    if len(tools) > limit:
        results["_truncated"] = f"{len(tools) - limit} additional tool request(s) dropped by the per-round cap"
    return results


def precondition_facts(stack: SandboxStack) -> dict[str, Any]:
    """The telemetry the action registry's preconditions read.

    Fetched by CODE at the guardrail, not requested by the model: "is there a
    healthy alternate to fail over to?" must be answered from fresh data, whatever
    the Diagnostician did or did not ask for.
    """
    return {"deploys": get_deploy_history(stack), "dependency_health": get_dependency_health(stack)}


def observe_all(stack: SandboxStack) -> dict[str, Any]:
    """The default telemetry bundle handed to every agent at the top of a pass.

    The internal `faulted` ground-truth flag is intentionally NOT exposed: agents
    must infer health from the actual metrics (error_rate, latency), exactly as a
    real on-call engineer would, rather than reading a boolean answer.
    """
    metrics = {k: v for k, v in get_metrics(stack).items() if k != "faulted"}
    return {
        "metrics": metrics,
        "recent_logs": get_logs(stack),
        "traces": get_traces(stack),
        "deploys": get_deploy_history(stack),
    }

"""Shared machinery for the five agents.

There is no framework here on purpose. A delegation is a function call that
takes a budget view and returns a schema-checked dict, and this file is all of
the plumbing that requires. If it looks small, that is the argument: the hard
parts of a multi-agent system are not the loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..budget import Delegation, RequestBudget
from ..failures import FailureSwitches
from ..llm import Usage
from ..prompts import Prompt
from ..trace import Span, Trace


@dataclass
class Ctx:
    """Everything an agent needs, in one envelope: the trace id, the budget and
    the tenant travel together, exactly as the post asks."""

    trace: Trace
    budget: RequestBudget
    switches: FailureSwitches
    client: Any
    tenant_id: str
    tenant_name: str
    cache_on: bool


async def call_model(
    ctx: Ctx,
    *,
    parent: Span,
    stage: int,
    agent_id: str,
    prompt: Prompt,
    prompt_key: str,
    model: str,
    facts: dict[str, Any],
    delegation: Delegation | None = None,
) -> tuple[dict[str, Any], Span]:
    """One model call, one span. The span records the model, the prompt version,
    the token split including cached tokens, and the budget state at entry."""
    span = ctx.trace.start(
        f"model:{agent_id}",
        stage,
        agent_id,
        parent=parent,
        kind="model",
        prompt_key=prompt_key,
        model=model,
        **(delegation.as_span_fields() if delegation else {}),
    )
    try:
        result, usage = await ctx.client.complete(
            prompt, model=model, facts=facts, cache_on=ctx.cache_on
        )
    except Exception as exc:  # noqa: BLE001
        span.finish("error", str(exc))
        raise

    _apply_usage(span, usage)
    (delegation or ctx.budget).spend(usage.input_tokens + usage.output_tokens)

    if usage.fell_back:
        span.warnings.append(f"top-tier model unavailable, fell back to {usage.model}")
    if result.get("error") == "schema_violation":
        # A handoff that does not validate is rejected at the boundary. The post
        # calls free-form handoffs the single most common source of quiet
        # failure, and this is the line that stops it being quiet.
        span.finish("error", "schema_violation")
    else:
        span.finish("ok")
    return result, span


def _apply_usage(span: Span, usage: Usage) -> None:
    span.model = usage.model
    span.input_tokens = usage.input_tokens
    span.cached_tokens = usage.cached_tokens
    span.output_tokens = usage.output_tokens
    span.detail["cache_hit_pct"] = (
        round(100 * usage.cached_tokens / usage.input_tokens) if usage.input_tokens else 0
    )


async def run_agent(
    ctx: Ctx,
    *,
    parent: Span,
    stage: int,
    agent_id: str,
    timeout_s: float,
    body: Callable[[Delegation, Span], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Run one sub-agent under its own timeout, which is always shorter than the
    whole-request deadline.

    A timeout here cuts one branch. It does not fail the request, because the
    other branches have results and a partial answer beats no answer.
    """
    span = ctx.trace.start(f"agent:{agent_id}", stage, agent_id, parent=parent, kind="stage")
    try:
        delegation = ctx.budget.delegate(agent_id, agent_timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - depth cap or exhausted budget
        span.finish("skipped", str(exc))
        return {"error": "budget_exceeded", "detail": str(exc)}

    for k, v in delegation.as_span_fields().items():
        setattr(span, k, v)

    try:
        result = await asyncio.wait_for(body(delegation, span), timeout=delegation.timeout_s)
    except asyncio.TimeoutError:
        span.finish("timeout", f"agent exceeded its {delegation.timeout_s:.1f}s slice")
        return {
            "error": "agent_timeout",
            "agent_id": agent_id,
            "timeout_s": round(delegation.timeout_s, 2),
        }
    except AgentKilled as exc:
        span.finish("killed", str(exc))
        return {"error": "agent_killed", "agent_id": agent_id, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        span.finish("error", str(exc))
        return {"error": "agent_error", "agent_id": agent_id, "detail": str(exc)}

    if result.get("error"):
        span.finish("error", str(result["error"]))
    else:
        span.finish("ok")
    return result


class AgentKilled(Exception):
    """Raised by the kill-the-shipping-agent toggle. Distinct from a crash so
    the trace can say what happened rather than showing a generic error."""


def compress_for_return(result: dict[str, Any], keep: list[str]) -> dict[str, Any]:
    """Compress on the way out.

    The supervisor's context window is the bottleneck, so a branch returns a
    summary and not its full working. The full output stays in the span, which
    is where you go when you need it.
    """
    return {k: result.get(k) for k in keep if k in result}

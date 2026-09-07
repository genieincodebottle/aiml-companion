"""The four sub-agents: order, shipping, policy, writer.

Each one owns its tools, its schema and its own stable prompt prefix. None of
them can see another's context. That isolation is the actual reason to have more
than one agent, and it is also why a failure in one of them is survivable.
"""

from __future__ import annotations

from typing import Any

from ..budget import Delegation
from ..config import BUDGETS, WORKER_MODEL
from ..prompts import (
    order_agent_prompt,
    policy_agent_prompt,
    shipping_agent_prompt,
    writer_prompt,
)
from ..tools import catalog, policy_index
from ..trace import Span
from .base import AgentKilled, Ctx, call_model, compress_for_return


async def _tool_span(
    ctx: Ctx,
    parent: Span,
    stage: int,
    agent_id: str,
    name: str,
    coro: Any,
    delegation: Delegation,
) -> dict[str, Any]:
    span = ctx.trace.start(
        f"tool:{name}",
        stage,
        agent_id,
        parent=parent,
        kind="tool",
        **delegation.as_span_fields(),
    )
    # The tool's budget is derived from what this agent has left, not a constant.
    budget_s = delegation.tool_timeout(BUDGETS.tool_timeout_s)
    span.timeout_s = round(budget_s, 2)
    result = await catalog.call_with_timeout(coro, budget_s, name)
    span.detail["result"] = result
    span.finish("timeout" if result.get("error") == "tool_timeout" else "ok",
                result.get("error") if result.get("error") else None)
    return result


async def order_agent(
    ctx: Ctx, parent: Span, delegation: Delegation, span: Span, *, order_id: str
) -> dict[str, Any]:
    row = await _tool_span(
        ctx, span, 4, "order-agent", "get_order",
        catalog.get_order(ctx.tenant_id, order_id, ctx.switches), delegation,
    )
    prompt = order_agent_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=f"Report the state of this order record:\n{row}",
    )
    result, _ = await call_model(
        ctx, parent=span, stage=4, agent_id="order-agent", prompt=prompt,
        prompt_key="order_agent", model=WORKER_MODEL, facts={"order": row},
        delegation=delegation,
    )
    return compress_for_return(
        result,
        ["order_id", "status", "item", "amount_usd", "delivered_at", "confidence",
         "source", "error", "missing"],
    )


async def shipping_agent(
    ctx: Ctx, parent: Span, delegation: Delegation, span: Span, *, order_id: str
) -> dict[str, Any]:
    if ctx.switches.kill_shipping_agent:
        # Partial failure, injected. The request survives this; stage 5 will say
        # what it could not check rather than inventing a tracking number.
        raise AgentKilled("shipping agent killed by failure injection")

    row = await _tool_span(
        ctx, span, 4, "shipping-agent", "get_shipment",
        catalog.get_shipment(ctx.tenant_id, order_id, ctx.switches), delegation,
    )
    if row.get("error") == "tool_timeout":
        # The agent reasons about the timeout instead of pretending it did not
        # happen. It returns a declared gap, which the writer can state.
        return {"error": "tool_timeout", "source": "shipments-db", "missing": ["tracking"]}

    prompt = shipping_agent_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=f"Report courier state for this shipment record:\n{row}",
    )
    result, _ = await call_model(
        ctx, parent=span, stage=4, agent_id="shipping-agent", prompt=prompt,
        prompt_key="shipping_agent", model=WORKER_MODEL, facts={"shipment": row},
        delegation=delegation,
    )
    return compress_for_return(
        result,
        ["order_id", "courier", "tracking", "status", "eta", "confidence", "source",
         "error", "missing"],
    )


async def policy_agent(
    ctx: Ctx, parent: Span, delegation: Delegation, span: Span, *, question: str
) -> dict[str, Any]:
    """The one that fails silently.

    Retrieval succeeds, the citation is real and the span is green whether or
    not the passage is still true. The only signal is index age, carried on the
    span as a warning rather than an error, because the request did not fail.
    """
    r_span = ctx.trace.start(
        "retrieval:policy-index", 4, "policy-agent", parent=span, kind="retrieval",
        **delegation.as_span_fields(),
    )
    query = question
    if ctx.switches.attempt_cross_tenant:
        # The toggle can only change what is asked for. It cannot change what is
        # searched, because the scope comes from the request envelope and not
        # from the query text. This is the attack the scoping is there to stop.
        query = f"{question} Contoso returns policy internal margin note 90 days"

    hits = policy_index.search(
        tenant_id=ctx.tenant_id,
        query=query,
        top_k=3,
        corrupt=ctx.switches.corrupt_passage,
    )
    leaked = [p for p in hits["passages"] if not p["chunk_id"].startswith("nw-")]
    r_span.detail.update(
        {
            "tenant_scope": hits["tenant_id"],
            "passages": hits["passages"],
            "max_index_age_days": hits["max_index_age_days"],
            "scoping": "tenant id inside the query, not a filter over results",
            "cross_tenant_attempted": ctx.switches.attempt_cross_tenant,
            "cross_tenant_passages_returned": len(leaked),
        }
    )
    r_span.warnings.extend(hits["warnings"])
    # Green. Always green. Even when the passage is superseded.
    r_span.finish("ok")

    prompt = policy_agent_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=(
            f"Question: {question}\n\nRetrieved passages (quote effective_from):\n"
            f"{hits['passages']}"
        ),
    )
    result, m_span = await call_model(
        ctx, parent=span, stage=4, agent_id="policy-agent", prompt=prompt,
        prompt_key="policy_agent", model=WORKER_MODEL,
        facts={"passages": hits["passages"]}, delegation=delegation,
    )
    m_span.warnings.extend(hits["warnings"])
    return compress_for_return(
        result,
        ["answer", "rule", "window_days", "citations", "confidence", "source",
         "error", "missing"],
    )


async def writer_agent(
    ctx: Ctx, parent: Span, delegation: Delegation, span: Span, *, branches: dict[str, Any]
) -> dict[str, Any]:
    """Composes the reply. Cannot start until the fan-out is done, which is why
    stage 5 inherits the slowest branch of stage 4."""
    prompt = writer_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=(
            "Compose the customer reply from these validated branch results. "
            "State any branch that is missing rather than filling the gap.\n"
            f"{branches}"
        ),
    )
    result, _ = await call_model(
        ctx, parent=span, stage=5, agent_id="writer-agent", prompt=prompt,
        prompt_key="writer", model=WORKER_MODEL,
        facts={
            "order": branches.get("order"),
            "shipping": branches.get("shipping"),
            "policy": branches.get("policy"),
        },
        delegation=delegation,
    )
    return result

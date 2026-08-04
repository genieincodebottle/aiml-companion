"""The eight stages, in order, written out explicitly.

    1  input guardrail
    2  classify and route
    3  orchestrator plan          top-tier model, first call
    4  fan-out                    order, shipping and policy agents in parallel
    5  writer agent               composes the reply
    6  orchestrator merge         top-tier model, second call
    7  output guardrail
    8  respond

Five agents total: the three lookups, the writer, and the orchestrator, which is
itself an agent rather than a scheduler.

These numbers appear in the span names, the log lines and the UI labels, and
they match the diagram in the README. If you renumber here, renumber there.

No agent framework. The fan-out below is asyncio.gather over three functions,
the delegation is an argument, and the loop is a for loop. That is the whole
argument of the project: a framework would hide both the bill and the failure
mode, and the failure modes are what this is for.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from . import guardrails
from .agents import orchestrator as orch
from .agents.base import Ctx, run_agent
from .agents.workers import order_agent, policy_agent, shipping_agent, writer_agent
from .budget import RequestBudget
from .config import BUDGETS, CLASSIFIER_MODEL, DEFAULT_TENANT
from .failures import FailureSwitches
from .llm import PREFIX_CACHE, estimate_tokens, get_client, price_for
from .tools import db
from .tools.saga import build_return_booking_saga
from .trace import Trace

TENANT_NAMES = {"tenant-northwind": "Northwind Home", "tenant-contoso": "Contoso Kitchen"}


async def run_request(
    question: str,
    *,
    tenant_id: str = DEFAULT_TENANT,
    customer_id: str = "cust-1001",
    switches: FailureSwitches | None = None,
    reset_cache: bool = False,
) -> dict[str, Any]:
    db.init_db()
    switches = switches or FailureSwitches()
    if reset_cache:
        # Lets the cost panel show a cold-cache request on demand.
        PREFIX_CACHE.reset()

    client = get_client()
    request_id, is_retry = guardrails.canonical_request_id(tenant_id, question)
    trace = Trace(request_id, tenant_id, client.name)

    budget = RequestBudget.start(
        deadline_s=BUDGETS.request_deadline_s,
        tokens=BUDGETS.request_token_budget,
        request_id=request_id,
        tenant_id=tenant_id,
    )
    ctx = Ctx(
        trace=trace,
        budget=budget,
        switches=switches,
        client=client,
        tenant_id=tenant_id,
        tenant_name=TENANT_NAMES.get(tenant_id, tenant_id),
        cache_on=not switches.disable_prompt_cache,
    )

    root = trace.start("request", 0, "gateway")
    root.detail.update(
        {
            "question": question,
            "canonical_request_id": request_id,
            "is_retry_of_same_request": is_retry,
            "failures_active": switches.active(),
            "mode": client.name,
        }
    )
    root.deadline_remaining_s = budget.remaining_s
    root.tokens_remaining = budget.tokens_remaining

    # ---------------- stage 1: input guardrail ----------------
    s1 = trace.start("stage-1:input-guardrail", 1, "gateway", parent=root)
    allowed, why = guardrails.RATE_LIMITER.check()
    screen = guardrails.screen_input(question)
    s1.detail.update({"screen": screen, "rate_limit_ok": allowed, "rate_limit_reason": why})
    s1.deadline_remaining_s = budget.remaining_s
    if not allowed or screen["blocked"]:
        s1.finish("error", why or "input blocked by guardrail")
        root.finish("error", "blocked at stage 1")
        return _respond(trace, "This request was blocked before any model was called.", [])
    s1.finish("ok")

    # ---------------- stage 2: classify and route ----------------
    s2 = trace.start(
        "stage-2:classify-and-route", 2, "gateway", parent=root, kind="model",
        prompt_key="classifier", model=CLASSIFIER_MODEL,
    )
    route = _classify(question)
    # The classifier is a real billed call, so it is costed like one. It is also
    # tiny, which is the point: it decides whether the expensive fan-out runs.
    s2.input_tokens = estimate_tokens(question) + 120
    s2.output_tokens = 2
    s2.detail.update({"route": route, "why": "cheap classification before any fan-out"})
    s2.deadline_remaining_s = budget.remaining_s
    s2.finish("ok")

    if route == "cached":
        root.finish("ok")
        return _respond(
            trace,
            "Hello. Ask me about an order, a delivery or a return and I will look it up.",
            [],
        )

    # ---------------- stage 3: orchestrator plan ----------------
    plan = await orch.plan(ctx, root, question=question, customer_id=customer_id)
    order_id = plan.get("order_id") or "ORD-4412"
    wanted = plan.get("branches") or ["order", "shipping", "policy"]
    if route == "single":
        wanted = wanted[:1]

    # ---------------- stage 4: fan-out ----------------
    fan = trace.start("stage-4:fan-out", 4, "orchestrator", parent=root)
    fan.deadline_remaining_s = budget.remaining_s
    fan.tokens_remaining = budget.tokens_remaining
    fan.detail["branches"] = wanted
    fan.detail["note"] = (
        "three independent lookups with no ordering between them; the fan-out "
        "inherits its slowest branch"
    )

    async def _order() -> dict[str, Any]:
        return await run_agent(
            ctx, parent=fan, stage=4, agent_id="order-agent",
            timeout_s=BUDGETS.order_agent_timeout_s,
            body=lambda d, s: order_agent(ctx, fan, d, s, order_id=order_id),
        )

    async def _shipping() -> dict[str, Any]:
        return await run_agent(
            ctx, parent=fan, stage=4, agent_id="shipping-agent",
            timeout_s=BUDGETS.shipping_agent_timeout_s,
            body=lambda d, s: shipping_agent(ctx, fan, d, s, order_id=order_id),
        )

    async def _policy() -> dict[str, Any]:
        return await run_agent(
            ctx, parent=fan, stage=4, agent_id="policy-agent",
            timeout_s=BUDGETS.policy_agent_timeout_s,
            body=lambda d, s: policy_agent(
                ctx, fan, d, s, question=plan.get("policy_question") or question
            ),
        )

    runners = {"order": _order, "shipping": _shipping, "policy": _policy}
    selected = [(n, runners[n]) for n in wanted if n in runners]

    # The fan-out itself. Three delegations, one gather, no framework.
    results = await asyncio.gather(*(fn() for _, fn in selected), return_exceptions=True)
    branches: dict[str, Any] = {}
    for (name, _), res in zip(selected, results):
        branches[name] = (
            {"error": "agent_exception", "detail": str(res)}
            if isinstance(res, BaseException)
            else res
        )

    # The saga runs inside stage 4 because it is a tool-side effect of the plan,
    # not a separate stage.
    if plan.get("needs_saga"):
        await _run_saga(ctx, fan, order_id=order_id, request_id=request_id)

    failed = [n for n, v in branches.items() if v.get("error")]
    fan.detail["failed_branches"] = failed
    # A fan-out with a dead branch is not a failed fan-out. It is a partial one,
    # and the request continues on what did return.
    fan.finish("ok" if not failed else "error", f"{len(failed)} branch(es) did not return" if failed else None)

    # ---------------- stage 5: writer ----------------
    w = trace.start("stage-5:writer", 5, "writer-agent", parent=root)
    w.deadline_remaining_s = budget.remaining_s
    w.tokens_remaining = budget.tokens_remaining
    draft = await run_agent(
        ctx, parent=w, stage=5, agent_id="writer-agent",
        timeout_s=BUDGETS.writer_agent_timeout_s,
        body=lambda d, s: writer_agent(ctx, w, d, s, branches=branches),
    )
    w.detail["gaps"] = draft.get("gaps", [])
    w.finish("ok" if not draft.get("error") else "error", draft.get("error"))

    # ---------------- stage 6: orchestrator merge ----------------
    merged = await orch.merge(ctx, root, draft=draft, branches=branches)
    reply = merged.get("final_reply") or draft.get("reply") or ""

    # ---------------- stage 7: output guardrail ----------------
    s7 = trace.start("stage-7:output-guardrail", 7, "gateway", parent=root)
    reply, redacted = guardrails.redact_output(reply)
    s7.detail.update({"redacted": redacted, "unresolved": merged.get("unresolved", [])})
    s7.deadline_remaining_s = budget.remaining_s
    s7.finish("ok")

    # ---------------- stage 8: respond ----------------
    s8 = trace.start("stage-8:respond", 8, "gateway", parent=root)
    s8.detail["confidence"] = merged.get("confidence")
    s8.finish("ok")
    root.finish("ok")

    guardrails.RATE_LIMITER.record_spend(trace.total_cost_usd)
    return _respond(trace, reply, draft.get("gaps", []), merged=merged, plan=plan)


async def _run_saga(ctx: Ctx, parent: Any, *, order_id: str, request_id: str) -> None:
    span = ctx.trace.start("saga:book-return", 4, "orchestrator", parent=parent, kind="saga")
    saga = build_return_booking_saga(
        request_id=request_id,
        tenant_id=ctx.tenant_id,
        order_id=order_id,
        fail_order_update=ctx.switches.fail_order_update,
    )
    results = await saga.run()
    for r in results:
        child = ctx.trace.start(
            f"{'undo:' if r.direction == 'compensate' else ''}{r.name.replace('undo:', '')}",
            4, "orchestrator", parent=span, kind="saga",
        )
        child.detail.update({"system": r.system, "direction": r.direction, "payload": r.payload})
        child.finish("error" if r.status == "error" else "ok", r.payload.get("error"))
    span.detail["rolled_back"] = saga.rolled_back
    span.detail["order"] = [f"{r.direction}:{r.name}" for r in results]
    span.finish("error" if saga.rolled_back else "ok",
                "step 3 failed, compensators ran in reverse" if saga.rolled_back else None)


_TOPICS = {
    "order": ("order", "purchase", "bought", "ord-"),
    "shipping": ("ship", "track", "deliver", "courier", "parcel", "where is"),
    "policy": ("refund", "return", "policy", "exchange", "money back"),
}


def _classify(question: str) -> str:
    """Stage 2, as a rule rather than a model call.

    A rule is the right implementation here: it costs nothing, it is
    deterministic, and the point of stage 2 is only to decide whether the
    expensive fan-out is justified. Swapping it for the cheap model in
    CLASSIFIER_PROMPT changes the cost line and nothing else.
    """
    q = question.strip().lower()
    if len(q) < 16 and any(g in q for g in ("hi", "hello", "hey", "thanks")):
        return "cached"
    topics = sum(1 for words in _TOPICS.values() if any(w in q for w in words))
    if topics >= 2:
        return "full"
    return "single" if topics == 1 else "full"


def _respond(
    trace: Trace,
    reply: str,
    gaps: list[str],
    *,
    merged: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings = [w for s in trace.spans for w in s.warnings]
    return {
        "reply": reply,
        "gaps": gaps,
        "plan": plan,
        "merge": merged,
        "trace": trace.to_dict(),
        # The honest bit. Spans green does not mean answer right, and the panel
        # says so out loud when a semantic warning was raised.
        "semantic_warnings": warnings,
        "all_spans_green": all(s.status == "ok" for s in trace.spans),
    }


def cost_breakdown(trace_dict: dict[str, Any]) -> dict[str, Any]:
    """Per-model rollup for the cost panel, from a finished trace."""
    by_model: dict[str, dict[str, Any]] = {}
    for s in trace_dict["spans"]:
        if not s["model"]:
            continue
        m = by_model.setdefault(
            s["model"],
            {"calls": 0, "input": 0, "cached": 0, "output": 0, "usd": 0.0,
             "price": price_for(s["model"]).__dict__},
        )
        m["calls"] += 1
        m["input"] += s["input_tokens"]
        m["cached"] += s["cached_tokens"]
        m["output"] += s["output_tokens"]
        m["usd"] += s["cost_usd"]
    for m in by_model.values():
        m["usd"] = round(m["usd"], 6)
    return by_model


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]

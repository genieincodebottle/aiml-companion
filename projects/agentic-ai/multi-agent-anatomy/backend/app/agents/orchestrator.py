"""The orchestrator: stage 3 (plan) and stage 6 (merge).

It is the fifth agent, not a scheduler. It runs on the top-tier model and it is
called exactly twice per request, which is the only reason a top-tier model is
affordable here.

Both calls share the same stable prefix, so the second one is largely a cache
hit. That is a deliberate consequence of the prompt layout in prompts.py, not a
coincidence.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import BUDGETS, ORCHESTRATOR_MODEL
from ..prompts import orchestrator_merge_prompt, orchestrator_plan_prompt
from ..trace import Span
from .base import Ctx, call_model

_ORDER_RE = re.compile(r"\bORD-\d+\b", re.I)


async def plan(ctx: Ctx, parent: Span, *, question: str, customer_id: str) -> dict[str, Any]:
    """Stage 3. Turn the request into a plan.

    The plan is checkpointed on the span before anything is delegated, so a
    request that dies in stage 4 still shows what it intended to do.
    """
    span = ctx.trace.start("stage-3:orchestrator-plan", 3, "orchestrator", parent=parent)
    span.deadline_remaining_s = ctx.budget.remaining_s
    span.tokens_remaining = ctx.budget.tokens_remaining

    order_id = (_ORDER_RE.search(question).group(0).upper() if _ORDER_RE.search(question) else None)

    prompt = orchestrator_plan_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=(
            f"Customer: {customer_id}\nRequest: {question}\n"
            f"Remaining deadline: {ctx.budget.remaining_s:.1f}s\n"
            f"Remaining token budget: {ctx.budget.tokens_remaining}\n"
            f"Max delegation depth: {BUDGETS.max_delegation_depth}\n"
        ),
    )
    result, _ = await call_model(
        ctx, parent=span, stage=3, agent_id="orchestrator", prompt=prompt,
        prompt_key="orchestrator_plan", model=ORCHESTRATOR_MODEL,
        facts={
            "order_id": order_id,
            "customer_id": customer_id,
            "branches": ["order", "shipping", "policy"],
            "policy_question": question,
            # A saga is only planned when the customer asks for the change to be
            # made. Asking whether a return is possible is a question; asking
            # for it to be booked is three writes across three systems.
            "needs_saga": bool(
                re.search(r"\b(book|process|arrange|start)\b", question, re.I)
                and re.search(r"\b(return|refund)\b", question, re.I)
            ),
        },
    )
    result.setdefault("order_id", order_id)

    # Plan checkpoint. Written after each step, so a retry resumes rather than
    # replanning from nothing.
    span.detail["plan"] = result
    span.detail["checkpoint"] = "plan-v1"
    span.finish("ok")
    return result


async def merge(
    ctx: Ctx, parent: Span, *, draft: dict[str, Any], branches: dict[str, Any]
) -> dict[str, Any]:
    """Stage 6. Second top-tier call.

    Merging is not rewriting. A branch that failed is listed as rejected and its
    gap is carried into `unresolved` rather than being smoothed over, because a
    confident sentence built on a missing branch is exactly the failure this
    project is about.
    """
    span = ctx.trace.start("stage-6:orchestrator-merge", 6, "orchestrator", parent=parent)
    span.deadline_remaining_s = ctx.budget.remaining_s
    span.tokens_remaining = ctx.budget.tokens_remaining

    prompt = orchestrator_merge_prompt(
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant_name,
        task=(
            f"Draft reply from the writer:\n{draft.get('reply', '')}\n\n"
            f"Branch results:\n{branches}\n\n"
            f"Declared gaps: {draft.get('gaps', [])}\n"
        ),
    )
    result, _ = await call_model(
        ctx, parent=span, stage=6, agent_id="orchestrator", prompt=prompt,
        prompt_key="orchestrator_merge", model=ORCHESTRATOR_MODEL,
        facts={"phase": "merge", "draft": draft.get("reply", ""), "branches": branches},
    )
    span.detail["accepted"] = result.get("accepted_branches")
    span.detail["rejected"] = result.get("rejected_branches")
    span.finish("ok")
    return result

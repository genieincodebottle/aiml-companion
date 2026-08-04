"""Prompt layout, ordered for the prompt cache.

Providers cache a *prefix*. The cache matches from the first token forward and
stops at the first byte that differs, so anything that changes per request
poisons everything behind it. The whole discipline is one rule:

    order every prompt from least changing to most changing

Each prompt below is assembled in four bands, and the boundaries are marked in
the code so the reason is visible at the point it matters:

    BAND 1  role and rules        identical for the life of the prompt version
    BAND 2  tool and schema docs  changes only on deploy
    BAND 3  tenant context        changes per tenant, stable across their requests
    ------- CACHE BOUNDARY -----  everything above is the stable prefix
    BAND 4  this request          the question, the plan, the branch results

Each agent gets its own stable prefix. They are not shared, because a shared
prefix would have to contain the union of five agents' rules and then none of
them would match cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

CACHE_BOUNDARY = "\n<<<CACHE BOUNDARY: everything below varies per request>>>\n"


@dataclass
class Prompt:
    """A prompt split at the cache boundary so token accounting can tell the
    cacheable part from the part that is billed fresh every time."""

    agent_id: str
    version: str
    stable_prefix: str
    variable_suffix: str

    @property
    def full(self) -> str:
        return self.stable_prefix + CACHE_BOUNDARY + self.variable_suffix


# --------------------------------------------------------------------------
# BAND 1: role and rules. Never contains a value from the current request.
# --------------------------------------------------------------------------

_SHARED_RULES = """\
You are one agent inside an order-support system for an ecommerce platform.

Rules that apply to you at all times:
- Answer only from the data given to you. If a fact is not in your input, say so.
- Never invent an order id, a tracking number, a date, an amount or a policy rule.
- Every claim you make must be traceable to a field you were given.
- If your input is incomplete, return what you have and state what is missing.
- Reply with JSON only. No prose outside the JSON object.
"""

_ORDER_ROLE = """\
You are the order agent. You look up order records and report their state.
You do not reason about shipping, and you do not quote policy.
"""

_SHIPPING_ROLE = """\
You are the shipping agent. You report courier state, tracking and estimated
delivery. You do not quote policy and you do not restate order totals.
"""

_POLICY_ROLE = """\
You are the policy agent. You answer only from retrieved policy passages.

You never hold the whole refund policy in your context. You are given the top
passages a retrieval step selected, and nothing else. This is the point of you:
context isolation. It is also your weakness, because nothing in the retrieval
step can tell you whether the passage you were handed is still true.

Therefore: every passage you are given carries an effective_from date. Quote it.
If the passage looks superseded, say so rather than answering confidently.
"""

_WRITER_ROLE = """\
You are the writer agent. You compose the customer-facing reply from the
structured results of the lookup agents.

You compose. You do not add facts. If a branch is missing or failed, say plainly
what could not be checked rather than filling the gap with a plausible sentence.
"""

_ORCHESTRATOR_ROLE = """\
You are the orchestrator. You turn a request into a plan, hand parts of it to
sub-agents, and decide when the work is done.

You have three lookup agents (order, shipping, policy) and one writer agent.
You run the three lookups in parallel and then the writer.

Hard limits you must respect:
- You may not delegate more than 2 levels deep.
- You may not run more than 3 planning iterations.
- You are given a remaining deadline and a remaining token budget. If either is
  short, cut the plan down rather than exceeding it. A partial answer inside
  budget is correct behaviour, not failure.
"""

# --------------------------------------------------------------------------
# BAND 2: tool and schema docs. Changes on deploy, not per request.
# --------------------------------------------------------------------------

_ORDER_SCHEMA = """\
Return this JSON shape:
{"order_id": str, "status": str, "item": str, "amount_usd": number,
 "placed_at": str, "delivered_at": str|null, "confidence": number,
 "source": "orders-db", "missing": [str]}
"""

_SHIPPING_SCHEMA = """\
Return this JSON shape:
{"order_id": str, "courier": str, "tracking": str, "status": str,
 "eta": str|null, "last_scan_at": str|null, "confidence": number,
 "source": "shipments-db", "missing": [str]}
"""

_POLICY_SCHEMA = """\
Return this JSON shape:
{"answer": str, "rule": str, "window_days": number|null,
 "citations": [{"chunk_id": str, "section": str, "effective_from": str}],
 "confidence": number, "source": "policy-index", "missing": [str]}

Provenance is part of the schema, not a nicety. A downstream agent cannot judge
your answer without knowing where it came from and how sure you were.
"""

_WRITER_SCHEMA = """\
Return this JSON shape:
{"reply": str, "gaps": [str], "used_branches": [str]}
"""

_ORCHESTRATOR_PLAN_SCHEMA = """\
Return this JSON shape:
{"intent": str, "order_id": str|null, "customer_id": str|null,
 "branches": ["order"|"shipping"|"policy"], "policy_question": str,
 "needs_saga": bool, "reason": str}
"""

_ORCHESTRATOR_MERGE_SCHEMA = """\
Return this JSON shape:
{"final_reply": str, "accepted_branches": [str], "rejected_branches": [str],
 "unresolved": [str], "confidence": number}

You are merging, not rewriting. If two branches disagree, say so in
`unresolved`. Do not pick a winner silently.
"""


def _tenant_band(tenant_id: str, tenant_name: str) -> str:
    """BAND 3: tenant context. Stable across every request from this tenant, so
    it belongs above the boundary. Putting it below would cost a full-price
    prefix on every request for no reason."""
    return (
        f"\nTenant: {tenant_id} ({tenant_name}).\n"
        "All data you see is scoped to this tenant. You never see another "
        "retailer's orders, shipments or policies, and you must not speculate "
        "about them.\n"
    )


def _prefix(*bands: str) -> str:
    return "\n".join(b.strip() for b in bands if b) + "\n"


def order_agent_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    return Prompt(
        agent_id="order-agent",
        version="v4",
        stable_prefix=_prefix(
            _ORDER_ROLE, _SHARED_RULES, _ORDER_SCHEMA, _tenant_band(tenant_id, tenant_name)
        ),
        variable_suffix=task,
    )


def shipping_agent_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    return Prompt(
        agent_id="shipping-agent",
        version="v4",
        stable_prefix=_prefix(
            _SHIPPING_ROLE, _SHARED_RULES, _SHIPPING_SCHEMA, _tenant_band(tenant_id, tenant_name)
        ),
        variable_suffix=task,
    )


def policy_agent_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    return Prompt(
        agent_id="policy-agent",
        version="v6",
        stable_prefix=_prefix(
            _POLICY_ROLE, _SHARED_RULES, _POLICY_SCHEMA, _tenant_band(tenant_id, tenant_name)
        ),
        # The retrieved passages go BELOW the boundary. They are the most
        # changing thing in the whole system: a different question retrieves
        # different chunks, so caching them would never hit.
        variable_suffix=task,
    )


def writer_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    return Prompt(
        agent_id="writer-agent",
        version="v5",
        stable_prefix=_prefix(
            _WRITER_ROLE, _SHARED_RULES, _WRITER_SCHEMA, _tenant_band(tenant_id, tenant_name)
        ),
        variable_suffix=task,
    )


def orchestrator_plan_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    return Prompt(
        agent_id="orchestrator",
        version="v7",
        stable_prefix=_prefix(
            _ORCHESTRATOR_ROLE,
            _SHARED_RULES,
            # Both schemas, deliberately. See orchestrator_merge_prompt below.
            _ORCHESTRATOR_PLAN_SCHEMA,
            _ORCHESTRATOR_MERGE_SCHEMA,
            _tenant_band(tenant_id, tenant_name),
        ),
        variable_suffix="Phase: plan.\n" + task,
    )


def orchestrator_merge_prompt(*, tenant_id: str, tenant_name: str, task: str) -> Prompt:
    """Stage 6 uses byte-identical bands 1 to 3 to stage 3.

    That is why both orchestrator schemas sit in the shared prefix even though
    each call only needs one of them. Splitting them would give the orchestrator
    two prefixes that both miss; keeping them together gives it one that hits on
    the second call of the same request. Which phase is running is stated below
    the boundary, where changing it costs nothing.
    """
    return Prompt(
        agent_id="orchestrator",
        version="v7",
        stable_prefix=_prefix(
            _ORCHESTRATOR_ROLE,
            _SHARED_RULES,
            _ORCHESTRATOR_PLAN_SCHEMA,
            _ORCHESTRATOR_MERGE_SCHEMA,
            _tenant_band(tenant_id, tenant_name),
        ),
        variable_suffix="Phase: merge.\n" + task,
    )


CLASSIFIER_PROMPT = """\
Classify an order-support request. Reply with one word.

full     needs order, shipping and policy lookups
single   needs one lookup only
cached   a greeting or a repeat of a question already answered

Cheap classification here is what decides whether a request needs the full
fan-out at all. A system that fans out to five agents for "hi" is paying six
model calls for nothing.

Request: """

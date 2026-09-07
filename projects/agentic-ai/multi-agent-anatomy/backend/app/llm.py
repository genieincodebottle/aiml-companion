"""The model layer: one provider, no framework.

Two clients implement the same call:

  LiveClient    talks to Gemini, needs GOOGLE_API_KEY or GEMINI_API_KEY
  ReplayClient  needs nothing at all, and is the default

ReplayClient is not a mock that returns a fixed string. It composes its answer
deterministically from the structured input it is handed, which is what makes
the failure toggles work with no key: corrupt the retrieved passage and the
replayed answer changes to the wrong refund window, exactly as the live model's
would. A mock that returned a canned reply would show a green trace and a
*correct* answer, which would teach the opposite of the point.

Token accounting is shared by both clients so the cost panel reads the same in
either mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import API_KEY, CACHE, FALLBACK_MODEL, live_mode_available, price_for
from .prompts import Prompt


@dataclass
class Usage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    model: str
    fell_back: bool = False


def estimate_tokens(text: str) -> int:
    """Roughly 4 characters per token. Good enough for a cost panel, and it is
    the same estimator on both sides so the two modes stay comparable."""
    return max(1, len(text) // 4)


class _PrefixCache:
    """Stands in for the provider's prompt cache so the cached-token counts in
    the trace move for a reason you can point at.

    A provider caches a prefix once it has seen it and once it is long enough.
    The first call in a process pays full price, every later call with the same
    prefix reports most of it as cached. Each agent has its own prefix, so they
    warm independently.
    """

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def reset(self) -> None:
        self._seen.clear()

    def lookup(self, prefix: str, *, enabled: bool) -> tuple[int, int]:
        """Return (prefix_tokens, cached_tokens)."""
        prefix_tokens = estimate_tokens(prefix)
        if not enabled:
            return prefix_tokens, 0
        key = hash(prefix)
        if prefix_tokens < CACHE.min_prefix_tokens:
            # Too short for the provider to bother caching. Real caches have a
            # minimum, and a project that ignores it overstates its savings.
            self._seen.add(key)
            return prefix_tokens, 0
        if key in self._seen:
            return prefix_tokens, int(prefix_tokens * CACHE.hit_ratio)
        self._seen.add(key)
        return prefix_tokens, 0


PREFIX_CACHE = _PrefixCache()


def _account(prompt: Prompt, output: str, model: str, *, cache_on: bool) -> Usage:
    prefix_tokens, cached = PREFIX_CACHE.lookup(prompt.stable_prefix, enabled=cache_on)
    suffix_tokens = estimate_tokens(prompt.variable_suffix)
    return Usage(
        input_tokens=prefix_tokens + suffix_tokens,
        cached_tokens=cached,
        output_tokens=estimate_tokens(output),
        model=model,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Models wrap JSON in fences often enough that this belongs in one place."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A schema violation is a real failure mode, not an edge case. The
        # caller turns this into a rejected branch rather than a crash.
        return {"error": "schema_violation", "raw": text[:400]}


class ReplayClient:
    """Deterministic. No network, no key, no nondeterminism."""

    name = "replay"

    async def complete(
        self, prompt: Prompt, *, model: str, facts: dict[str, Any], cache_on: bool
    ) -> tuple[dict[str, Any], Usage]:
        out = _compose(prompt.agent_id, facts)
        text = json.dumps(out)
        return out, _account(prompt, text, model, cache_on=cache_on)


class LiveClient:
    """One provider, called directly. No agent framework in between, because a
    framework would hide both the bill and the failure mode."""

    name = "live"

    def __init__(self) -> None:
        from google import genai  # imported lazily so replay mode needs no SDK

        self._client = genai.Client(api_key=API_KEY)

    async def complete(
        self, prompt: Prompt, *, model: str, facts: dict[str, Any], cache_on: bool
    ) -> tuple[dict[str, Any], Usage]:
        fell_back = False
        try:
            resp = await self._client.aio.models.generate_content(
                model=model, contents=prompt.full
            )
        except Exception:  # noqa: BLE001
            # Model ids get renamed. Falling back and recording it in the trace
            # beats a 500 that tells the reader nothing.
            resp = await self._client.aio.models.generate_content(
                model=FALLBACK_MODEL, contents=prompt.full
            )
            model, fell_back = FALLBACK_MODEL, True

        text = getattr(resp, "text", "") or ""
        usage = _account(prompt, text, model, cache_on=cache_on)
        usage.fell_back = fell_back

        # Prefer the provider's own counts when it gives them.
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage.input_tokens = getattr(meta, "prompt_token_count", usage.input_tokens)
            usage.output_tokens = getattr(meta, "candidates_token_count", usage.output_tokens)
            cached = getattr(meta, "cached_content_token_count", None)
            if cached is not None:
                usage.cached_tokens = cached if cache_on else 0
        return _extract_json(text), usage


def get_client() -> ReplayClient | LiveClient:
    if live_mode_available():
        try:
            return LiveClient()
        except Exception:  # noqa: BLE001 - a missing SDK must not break replay
            return ReplayClient()
    return ReplayClient()


# --------------------------------------------------------------------------
# The replay composer. One branch per agent, each reading the real tool output
# it was given. This is what keeps replay mode honest.
# --------------------------------------------------------------------------


def _compose(agent_id: str, facts: dict[str, Any]) -> dict[str, Any]:
    if agent_id == "orchestrator":
        return _compose_orchestrator(facts)
    if agent_id == "order-agent":
        return _compose_order(facts)
    if agent_id == "shipping-agent":
        return _compose_shipping(facts)
    if agent_id == "policy-agent":
        return _compose_policy(facts)
    if agent_id == "writer-agent":
        return _compose_writer(facts)
    return {"error": "unknown_agent", "agent_id": agent_id}


def _compose_orchestrator(facts: dict[str, Any]) -> dict[str, Any]:
    if facts.get("phase") == "merge":
        branches = facts.get("branches", {})
        accepted = [k for k, v in branches.items() if v and not v.get("error")]
        rejected = [k for k, v in branches.items() if not v or v.get("error")]
        unresolved = []
        for name in rejected:
            unresolved.append(f"{name} branch did not return")
        return {
            "final_reply": facts.get("draft", ""),
            "accepted_branches": accepted,
            "rejected_branches": rejected,
            "unresolved": unresolved,
            "confidence": round(0.95 - 0.2 * len(rejected), 2),
        }
    return {
        "intent": facts.get("intent", "order_status_and_refund"),
        "order_id": facts.get("order_id"),
        "customer_id": facts.get("customer_id"),
        "branches": facts.get("branches", ["order", "shipping", "policy"]),
        "policy_question": facts.get("policy_question", "refund window after delivery"),
        "needs_saga": facts.get("needs_saga", False),
        "reason": "three independent lookups, no ordering between them, so fan out",
    }


def _compose_order(facts: dict[str, Any]) -> dict[str, Any]:
    row = facts.get("order", {})
    if row.get("error"):
        return {"error": row["error"], "confidence": 0.0, "source": "orders-db", "missing": ["order"]}
    return {
        "order_id": row.get("order_id"),
        "status": row.get("status"),
        "item": row.get("item"),
        "amount_usd": row.get("amount_usd"),
        "placed_at": row.get("placed_at"),
        "delivered_at": row.get("delivered_at"),
        "confidence": 0.99,
        "source": "orders-db",
        "missing": [],
    }


def _compose_shipping(facts: dict[str, Any]) -> dict[str, Any]:
    row = facts.get("shipment", {})
    if row.get("error"):
        return {
            "error": row["error"],
            "confidence": 0.0,
            "source": "shipments-db",
            "missing": ["shipment"],
        }
    return {
        "order_id": row.get("order_id"),
        "courier": row.get("courier"),
        "tracking": row.get("tracking"),
        "status": row.get("status"),
        "eta": row.get("eta"),
        "last_scan_at": row.get("last_scan_at"),
        "confidence": 0.97,
        "source": "shipments-db",
        "missing": [],
    }


def _compose_policy(facts: dict[str, Any]) -> dict[str, Any]:
    passages = facts.get("passages", [])
    if not passages:
        return {
            "answer": "No policy passage matched this question.",
            "rule": None,
            "window_days": None,
            "citations": [],
            "confidence": 0.0,
            "source": "policy-index",
            "missing": ["policy passage"],
        }
    window = None
    for p in passages:
        m = re.search(r"within (\d+) days", p["text"])
        if m:
            window = int(m.group(1))
            break
    top = passages[0]
    return {
        # This reads the window straight out of the retrieved passage. Corrupt
        # the passage and this number is wrong, with nothing else changing.
        "answer": (
            f"Returns are accepted within {window} days of delivery."
            if window
            else top["text"]
        ),
        "rule": top["section"],
        "window_days": window,
        "citations": [
            {
                "chunk_id": p["chunk_id"],
                "section": p["section"],
                "effective_from": p["effective_from"],
            }
            for p in passages
        ],
        "confidence": 0.93,
        "source": "policy-index",
        "missing": [],
    }


def _compose_writer(facts: dict[str, Any]) -> dict[str, Any]:
    order = facts.get("order") or {}
    shipping = facts.get("shipping") or {}
    policy = facts.get("policy") or {}
    gaps: list[str] = []
    used: list[str] = []
    lines: list[str] = []

    if order and not order.get("error"):
        used.append("order")
        lines.append(
            f"Your order {order.get('order_id')} for the {order.get('item')} "
            f"({order.get('amount_usd')} USD) is currently {order.get('status')}."
        )
        if order.get("delivered_at"):
            lines.append(f"It was delivered on {order['delivered_at'][:10]}.")
    else:
        gaps.append("order details could not be retrieved")

    if shipping and not shipping.get("error"):
        used.append("shipping")
        lines.append(
            f"{shipping.get('courier')} has it under tracking {shipping.get('tracking')}, "
            f"status {shipping.get('status')}"
            + (f", estimated {shipping.get('eta')}." if shipping.get("eta") else ".")
        )
    else:
        # The partial-failure sentence. It states the gap rather than filling it.
        gaps.append("live courier tracking was unavailable for this reply")
        lines.append(
            "I could not reach the courier system just now, so I have not "
            "included live tracking. Everything else below is confirmed."
        )

    if policy and not policy.get("error"):
        used.append("policy")
        window = policy.get("window_days")
        if window:
            lines.append(
                f"On returns: you can return this within {window} days of delivery."
            )
        cites = policy.get("citations") or []
        if cites:
            lines.append(
                f"(Source: {cites[0]['section']}, effective {cites[0]['effective_from']}.)"
            )
    else:
        gaps.append("refund policy could not be checked")

    return {"reply": " ".join(lines), "gaps": gaps, "used_branches": used}


__all__ = [
    "Usage",
    "estimate_tokens",
    "get_client",
    "price_for",
    "PREFIX_CACHE",
    "ReplayClient",
    "LiveClient",
]

"""Failure injection.

This is the reason the project exists. Every toggle here maps to a named failure
mode in the post, and each one is meant to be turned on while you watch the
trace, not read about.

The fourth one is the important one. `corrupt_passage` produces a request in
which every span is green, the citation is real, the latency is good, the cost
is normal, and the answer is wrong. There is no status code for wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureSwitches:
    # Partial failure. The shipping agent dies; the answer still goes out built
    # from the two branches that did return, with the gap declared.
    kill_shipping_agent: bool = False

    # The fan-out inherits its slowest agent. Twelve seconds against an eight
    # second per-agent timeout means the agent is cut, not the request.
    slow_tool_seconds: float = 0.0
    slow_tool_target: str = "get_shipment"

    # Multi-step side effects with no transaction across them. Book the courier,
    # charge the fee, update the order; step three fails and the undos run in
    # reverse.
    fail_order_update: bool = False

    # The one that matters. A retrieved passage is silently replaced with an
    # outdated version of the refund window. Retrieval succeeds. The citation is
    # real. The span is green. The number in the answer is wrong.
    corrupt_passage: bool = False

    # Prompt caching off, so the cost panel can show the same request both ways.
    disable_prompt_cache: bool = False

    # Attempt a cross-tenant retrieval, to demonstrate that scoping is inside
    # the query and the attempt returns nothing rather than being filtered late.
    attempt_cross_tenant: bool = False

    def active(self) -> list[str]:
        out: list[str] = []
        if self.kill_shipping_agent:
            out.append("kill_shipping_agent")
        if self.slow_tool_seconds > 0:
            out.append(f"slow_tool:{self.slow_tool_target}:{self.slow_tool_seconds}s")
        if self.fail_order_update:
            out.append("fail_order_update")
        if self.corrupt_passage:
            out.append("corrupt_passage")
        if self.disable_prompt_cache:
            out.append("disable_prompt_cache")
        if self.attempt_cross_tenant:
            out.append("attempt_cross_tenant")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FailureSwitches":
        data = data or {}
        return cls(
            kill_shipping_agent=bool(data.get("kill_shipping_agent", False)),
            slow_tool_seconds=float(data.get("slow_tool_seconds", 0) or 0),
            slow_tool_target=str(data.get("slow_tool_target", "get_shipment")),
            fail_order_update=bool(data.get("fail_order_update", False)),
            corrupt_passage=bool(data.get("corrupt_passage", False)),
            disable_prompt_cache=bool(data.get("disable_prompt_cache", False)),
            attempt_cross_tenant=bool(data.get("attempt_cross_tenant", False)),
        )


CATALOG: list[dict[str, Any]] = [
    {
        "key": "kill_shipping_agent",
        "label": "Kill the shipping agent",
        "stage": 4,
        "type": "bool",
        "teaches": "Partial failure. The answer still goes out, built from what did return, with the gap stated rather than papered over.",
        "watch": "Stage 4 shows one red branch and two green. Stage 5 declares the missing branch. The request does not fail.",
    },
    {
        "key": "slow_tool_seconds",
        "label": "Add 12s latency to get_shipment",
        "stage": 4,
        "type": "seconds",
        "default_on": 12.0,
        "teaches": "A fan-out inherits its slowest agent. Without a per-agent timeout below the request deadline, one slow tool holds the whole request.",
        "watch": "The shipping branch hits its 8s timeout and is cut. Compare the fan-out duration against the sum of its children.",
    },
    {
        "key": "fail_order_update",
        "label": "Fail step 3 of the booking saga",
        "stage": 4,
        "type": "bool",
        "teaches": "No transaction spans the courier, the payment system and the order store. Every step needs its own undo, written in the same change as the step.",
        "watch": "Stage 4 runs book_courier, charge_fee, update_order. Step 3 fails and the undos run in reverse: refund_fee, then cancel_courier.",
    },
    {
        "key": "corrupt_passage",
        "label": "Serve a stale refund policy passage",
        "stage": 4,
        "type": "bool",
        "teaches": "The most dangerous failure in the post. Every signal says the system is healthy. Retrieval succeeded, the citation is real, the trace is green, and the answer is wrong.",
        "watch": "Nothing turns red. Compare the refund window in the answer against the policy document. Index age is the only metric that catches this.",
    },
    {
        "key": "disable_prompt_cache",
        "label": "Turn prompt caching off",
        "stage": 0,
        "type": "bool",
        "teaches": "The stable prefix of every agent is re-billed at full input price on every call.",
        "watch": "Cached tokens fall to zero across all spans and the running total rises.",
    },
    {
        "key": "attempt_cross_tenant",
        "label": "Attempt a cross-tenant retrieval",
        "stage": 4,
        "type": "bool",
        "teaches": "The tenant id goes inside the search query. Searching everything and dropping the wrong rows afterwards is a breach whether or not it reaches the answer.",
        "watch": "The retrieval span returns zero passages from the other tenant. Nothing was fetched and then discarded.",
    },
]

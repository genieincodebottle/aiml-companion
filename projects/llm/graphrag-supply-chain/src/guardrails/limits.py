"""Budget and resource guardrails.

The failure these prevent is not a crash, it is an invoice. An agentic or
retrieval system with no cap will happily spend real money on a malformed
question, a retry storm, or a loop nobody noticed, and the first signal is the
bill at the end of the month.

Four limits, each protecting a different resource:

  request budget   money and tokens, per request
  session budget   money and tokens, across a process or a user session
  rate limit       requests per window, per caller
  traversal depth  database work per query  (enforced in graph/queries.py,
                   because that is where it can be enforced *before* the query
                   is built)

A note on where limits belong. It is tempting to put a spend cap in the LLM
client, where it would be one method. It is put here, above the client, because
a budget is a policy decision and the client is a transport. Policy that lives
inside a transport cannot be varied per caller, tested without a network stub,
or reasoned about without reading the transport.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when a request or session exceeds its allowance.

    Deliberately an exception rather than a truncation. Silently returning a
    degraded answer when the budget runs out means the caller cannot tell a
    complete answer from a truncated one, which is a correctness problem
    wearing a cost-control costume.
    """


@dataclass
class Budget:
    max_llm_calls: int = 12
    max_input_tokens: int = 250_000
    max_output_tokens: int = 20_000
    max_usd: float = 0.50

    def check(self, usage: dict[str, Any], scope: str) -> None:
        if usage["llm_calls"] > self.max_llm_calls:
            raise BudgetExceeded(
                f"{scope} budget: {usage['llm_calls']} LLM calls exceeds the "
                f"limit of {self.max_llm_calls}."
            )
        if usage["input_tokens"] > self.max_input_tokens:
            raise BudgetExceeded(
                f"{scope} budget: {usage['input_tokens']} input tokens exceeds "
                f"the limit of {self.max_input_tokens}."
            )
        if usage["estimated_usd"] > self.max_usd:
            raise BudgetExceeded(
                f"{scope} budget: estimated ${usage['estimated_usd']:.4f} "
                f"exceeds the limit of ${self.max_usd:.2f}."
            )


class RateLimiter:
    """Fixed-window-free sliding window limiter, per caller key.

    In-process and therefore per-instance. That is correct for this project and
    wrong for production behind more than one worker: two replicas means two
    independent windows and twice the allowance. A real deployment moves this to
    Redis or to the ingress. Noted here rather than pretended away, because an
    in-memory limiter that a reader assumes is distributed is worse than no
    limiter at all.
    """

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, caller: str) -> tuple[bool, float]:
        """Returns (allowed, seconds_until_next_slot)."""
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(caller, deque())
            while window and now - window[0] > self.window:
                window.popleft()
            if len(window) >= self.max_requests:
                return False, self.window - (now - window[0])
            window.append(now)
            return True, 0.0

    def reset(self, caller: str | None = None) -> None:
        with self._lock:
            if caller is None:
                self._hits.clear()
            else:
                self._hits.pop(caller, None)


@dataclass
class InputLimits:
    """Caps on what a caller may send.

    `max_question_chars` is not politeness. An unbounded question is an
    unbounded prompt, an unbounded embedding call, and a way to push the system
    prompt out of the model's effective attention - which is a cheap and
    reliable jailbreak technique on its own, without any injection payload.
    """

    max_question_chars: int = 2_000
    min_question_chars: int = 3
    max_documents: int = 500
    max_document_chars: int = 200_000

    def check_question(self, question: str) -> None:
        text = (question or "").strip()
        if len(text) < self.min_question_chars:
            raise ValueError("The question is empty or too short to retrieve for.")
        if len(text) > self.max_question_chars:
            raise ValueError(
                f"The question is {len(text)} characters; the limit is "
                f"{self.max_question_chars}. Long inputs are capped because an "
                "unbounded prompt can push the system instructions out of the "
                "model's effective attention."
            )

    def check_document(self, text: str, doc_id: str) -> None:
        if len(text) > self.max_document_chars:
            raise ValueError(
                f"{doc_id} is {len(text)} characters, over the "
                f"{self.max_document_chars} limit. Split it, or raise the limit "
                "deliberately - document-level extraction has a context ceiling "
                "and silently truncating here would drop entities without an "
                "error."
            )


class UsageTracker:
    """Watches an LLMClient's cumulative usage against a session budget."""

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self.checkpoints: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[assignment]
        self.checkpoints = []

    def enforce(self, usage: dict[str, Any], scope: str = "session") -> None:
        self.checkpoints.append({"scope": scope, **usage})
        self.budget.check(usage, scope)

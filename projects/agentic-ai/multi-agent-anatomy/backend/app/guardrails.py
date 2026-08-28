"""Stage 1 (input) and stage 7 (output) guardrails.

Scope note: this project implements the two guardrails that sit on the request
path and nothing more. The wider security and governance material in the post is
deliberately out of scope here, because it does not get clearer by being
runnable.

What is in scope, because it is visible in the trace:
  - cost-aware rate limiting, not count-only
  - a canonical request id, so a retry is not a second fan-out
  - injection screening, honestly labelled as catching the obvious attempts
  - output redaction, so a customer email never leaves in a reply
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above)",
    r"disregard your (instructions|rules|system prompt)",
    r"reveal (your )?(system )?prompt",
    r"you are now",
    r"print (the )?(api[_ ]?key|secret|env)",
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


@dataclass
class RateLimiter:
    """Two limiters, because limiting by count alone lets a handful of expensive
    fan-outs through while blocking a flood of cheap ones."""

    max_requests_per_min: int = 30
    max_usd_per_min: float = 0.50
    _requests: list[float] = field(default_factory=list)
    _spend: list[tuple[float, float]] = field(default_factory=list)

    def _prune(self, now: float) -> None:
        self._requests = [t for t in self._requests if now - t < 60]
        self._spend = [(t, c) for t, c in self._spend if now - t < 60]

    def check(self) -> tuple[bool, str]:
        now = time.monotonic()
        self._prune(now)
        if len(self._requests) >= self.max_requests_per_min:
            return False, "rate limit: requests per minute"
        if sum(c for _, c in self._spend) >= self.max_usd_per_min:
            return False, "rate limit: spend per minute"
        self._requests.append(now)
        return True, ""

    def record_spend(self, usd: float) -> None:
        self._spend.append((time.monotonic(), usd))


RATE_LIMITER = RateLimiter()

# Canonical request ids. The same question from the same tenant inside the
# window is the same request, so a retry storm does not multiply the fan-out.
_SEEN: dict[str, float] = {}


def canonical_request_id(tenant_id: str, question: str) -> tuple[str, bool]:
    key = hashlib.sha256(f"{tenant_id}|{question.strip().lower()}".encode()).hexdigest()[:16]
    now = time.monotonic()
    for k, t in list(_SEEN.items()):
        if now - t > 120:
            _SEEN.pop(k, None)
    is_retry = key in _SEEN
    _SEEN[key] = now
    return key, is_retry


def screen_input(question: str) -> dict[str, object]:
    """Stage 1. Honest about its limits: pattern screening catches the obvious
    attempts and nothing subtler, and pretending otherwise is worse than not
    having it."""
    hits = [p for p in _INJECTION_PATTERNS if re.search(p, question, re.I)]
    too_long = len(question) > 4000
    return {
        "blocked": bool(hits) or too_long,
        "injection_patterns_hit": hits,
        "length_ok": not too_long,
        "coverage": "obvious patterns only; not a substitute for least privilege on tools",
    }


def redact_output(text: str) -> tuple[str, list[str]]:
    """Stage 7. What leaves is not what was generated."""
    found: list[str] = []
    if _EMAIL_RE.search(text):
        found.append("email")
        text = _EMAIL_RE.sub("[redacted email]", text)
    if _CARD_RE.search(text):
        found.append("card-like number")
        text = _CARD_RE.sub("[redacted number]", text)
    return text, found

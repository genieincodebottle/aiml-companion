"""Shared adapter plumbing: retries, prompt-block parsing, cost tables.

Kept out of ``__init__`` so the public surface of the package stays the
Protocol and nothing else.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, TypeVar

from . import is_retryable

log = logging.getLogger(__name__)

T = TypeVar("T")

# Published per-1M-token rates, used only to put an order-of-magnitude number in
# front of the learner. Vendors change prices; nobody updates a constant in a
# teaching repo the same week. Treat every figure this produces as an estimate,
# and never as a bill.
PRICES: dict[str, tuple[float, float]] = {
    # model-id prefix: (usd per 1M input, usd per 1M output)
    "gemini-3.5-flash": (0.30, 2.50),
    # Alias entries. Longest-prefix would be nicer, but the loop below takes the
    # first match, so the Flash entries must precede the broader "gemini-" one.
    "gemini-3.8-flash": (0.30, 2.50),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-pro-latest": (1.25, 10.00),
    "gemini-3.1-pro": (1.25, 10.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-": (2.50, 10.00),
    "claude-": (3.00, 15.00),
    "stub": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    for prefix, rates in PRICES.items():
        if model.startswith(prefix):
            return rates
    # Unknown model: zero rather than a guess. A wrong cost estimate presented
    # confidently is worse than a visibly absent one, because a learner will
    # believe it and size their retry budget against fiction.
    return (0.0, 0.0)


def with_retry(fn: Callable[[], T], *, attempts: int = 4, what: str = "call") -> T:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - vendors raise many types
            last = exc
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = 2**attempt
            log.warning(
                "%s failed (%s); retrying in %ss", what, type(exc).__name__, delay
            )
            time.sleep(delay)
    raise RuntimeError("unreachable retry loop exit") from last


# ---------------------------------------------------------------------------
# Prompt blocks
#
# Every prompt this project builds fences its structured payloads:
#
#     === RECORD (JSON) ===
#     {"id": "...", ...}
#     === END RECORD ===
#
# That is ordinary good prompt hygiene - an explicit boundary beats hoping the
# model infers where the data stops - and it has a second use here: the offline
# `stub` provider parses these blocks instead of guessing at prose. The stub can
# therefore be a real rule engine over real inputs rather than a canned reply,
# which is what lets all four phases run with no API key and still fail
# honestly.
# ---------------------------------------------------------------------------

_BLOCK_RE = "=== {name} \\(JSON\\) ===\\s*(.*?)\\s*=== END {name} ==="


def block(name: str, payload: Any) -> str:
    name = name.upper()
    body = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    return f"=== {name} (JSON) ===\n{body}\n=== END {name} ==="


def read_block(prompt: str, name: str) -> Any | None:
    match = re.search(
        _BLOCK_RE.format(name=re.escape(name.upper())), prompt, re.DOTALL
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def text_block(name: str, body: str) -> str:
    name = name.upper()
    return f"=== {name} ===\n{body}\n=== END {name} ==="


def read_text_block(prompt: str, name: str) -> str | None:
    n = re.escape(name.upper())
    match = re.search(f"=== {n} ===\\s*(.*?)\\s*=== END {n} ===", prompt, re.DOTALL)
    return match.group(1) if match else None

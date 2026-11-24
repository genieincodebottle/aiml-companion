"""PII detection + masking at the gateway boundary.

Prompts are scanned before they leave the process. Detected PII is replaced with
stable typed placeholders (``[EMAIL_1]``) so the model still sees co-reference
structure ("the same email appears twice") without ever seeing the raw value. The
mapping is kept per-call so a caller can re-hydrate placeholders in the response if
it must - but the *provider* and the audit log only ever see masked text.

This is deliberately regex-based and transparent. A production deployment swaps in a
dedicated PII service (Presidio, a cloud DLP API) behind this same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered so more specific patterns (SSN) win before generic number runs.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("PHONE", re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")),
]


@dataclass
class RedactionResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original

    @property
    def count(self) -> int:
        return len(self.mapping)


def redact(text: str) -> RedactionResult:
    """Replace PII with stable placeholders. Idempotent per value within a call."""
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}  # original -> placeholder (stable within a call)
    counters: dict[str, int] = {}

    def _sub(kind: str, match: re.Match[str]) -> str:
        original = match.group(0)
        if original in reverse:
            return reverse[original]
        counters[kind] = counters.get(kind, 0) + 1
        placeholder = f"[{kind}_{counters[kind]}]"
        mapping[placeholder] = original
        reverse[original] = placeholder
        return placeholder

    out = text
    for kind, pattern in _PATTERNS:
        out = pattern.sub(lambda m, k=kind: _sub(k, m), out)
    return RedactionResult(text=out, mapping=mapping)

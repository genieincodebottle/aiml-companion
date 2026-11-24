"""Semantic response cache.

Two requests that are *semantically* the same (not just byte-identical) should not
both pay for a model call. The cache embeds each prompt, and on lookup returns a
stored response when the nearest cached prompt exceeds a cosine-similarity threshold.

The embedding here is a deterministic hashing bag-of-words vector - no network, no
key, works in stub mode - but it is a genuine vector with genuine cosine similarity,
so the caching *behaviour* (near-duplicate hit, novel-prompt miss) is real and
testable. Swap `_embed` for a real embedding model and nothing else changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _embed(text: str) -> list[float]:
    """A deterministic hashed bag-of-words embedding (L2-normalized)."""
    vec = [0.0] * _DIM
    for tok in _TOKEN.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class _Entry:
    embedding: list[float]
    key: str  # exact routing key (model+tier) so we never cross-serve models
    response: str


@dataclass
class SemanticCache:
    threshold: float = 0.95
    _entries: list[_Entry] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def lookup(self, prompt: str, key: str) -> str | None:
        emb = _embed(prompt)
        best: tuple[float, _Entry] | None = None
        for entry in self._entries:
            if entry.key != key:
                continue
            sim = _cosine(emb, entry.embedding)
            if best is None or sim > best[0]:
                best = (sim, entry)
        if best and best[0] >= self.threshold:
            self.hits += 1
            return best[1].response
        self.misses += 1
        return None

    def store(self, prompt: str, key: str, response: str) -> None:
        self._entries.append(_Entry(embedding=_embed(prompt), key=key, response=response))

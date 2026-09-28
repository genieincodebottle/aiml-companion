"""Migration-rules retrieval tool (the RAG surface).

The worker calls `search_migration_rules(query)` to pull the guidance for the change
it has been asked to make. Retrieval here is deliberately embedding-free - a keyword
overlap score over the rulebook - so the demo has no vector-store dependency, but it
is structured as a retriever: swap `_score` for an embedding similarity and point it
at a real index and nothing else changes.

RAG is "just a tool" the agent decides when to call. That is the point: retrieval is
not welded into the loop, the agent reaches for it when it needs the guidance.
"""

from __future__ import annotations

import re

from ..rulebook.rules import RULES


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}


def _score(query_tokens: set[str], doc_text: str) -> float:
    doc = _tokens(doc_text)
    if not doc:
        return 0.0
    overlap = len(query_tokens & doc)
    return overlap / (len(query_tokens) ** 0.5 + 1e-9)


def search_migration_rules(worktree=None, query: str = "", top_k: int = 3) -> dict:
    """Return the top-k rules matching the query, best first. `worktree` is accepted
    (and ignored) so this matches the uniform tool handler signature."""
    q = _tokens(query)
    ranked = sorted(
        (
            {
                "id": rule.id,
                "name": rule.name,
                "summary": rule.summary,
                "guidance": rule.guidance,
                "score": round(_score(q, rule.retrieval_text), 4),
            }
            for rule in RULES.values()
        ),
        key=lambda r: r["score"],
        reverse=True,
    )
    hits = [r for r in ranked if r["score"] > 0][:top_k] or ranked[:1]
    return {"query": query, "results": hits, "count": len(hits)}

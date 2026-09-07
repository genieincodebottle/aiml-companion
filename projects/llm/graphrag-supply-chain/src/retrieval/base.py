"""Shared retrieval types and the fusion function.

Everything a strategy returns is a ``RetrievalResult``.  Keeping one shape for
all four strategies is what makes the comparison in the UI honest: the same
answer prompt, the same citation format, the same measurement, and the only
variable is how the evidence was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """One piece of retrieved context.

    ``kind`` is the important field.  Text evidence is a quote from a document.
    Graph evidence is a *derived* fact - a path the database computed, which
    appears verbatim in no document.  The answer layer renders and cites them
    differently, because they warrant different kinds of trust: a quote can be
    checked by reading, a derived fact can only be checked by re-running the
    traversal, so the traversal is shown.
    """

    kind: str                      # 'text' | 'graph_fact'
    text: str
    source_id: str                 # chunk_id, or a synthetic id for graph facts
    doc_id: str = ""
    title: str = ""
    doc_type: str = ""
    score: float = 0.0
    retrieved_by: str = ""         # 'vector' | 'keyword' | 'graph' | 'fused'
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.kind == "graph_fact":
            return f"[graph: {self.source_id}]"
        return f"[{self.doc_id}]"


@dataclass
class GraphEntity:
    key: str
    name: str
    type: str
    hops: int = 0
    score: float = 0.0
    path_names: list[str] = field(default_factory=list)
    path_rels: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    strategy: str
    evidence: list[Evidence] = field(default_factory=list)
    entities: list[GraphEntity] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    cypher_run: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    def context(self, max_chars: int) -> str:
        """Render evidence into the string the LLM sees.

        Graph facts go first.  Position matters: models attend more reliably to
        the start of a long context, and the derived facts are both the shortest
        and the highest-value part of what we retrieved.  Burying a three-line
        exposure path under 8,000 characters of supplier prose is a measurable
        way to lose the answer you did the traversal to get.
        """
        blocks: list[str] = []
        used = 0
        ordered = (
            [e for e in self.evidence if e.kind == "graph_fact"]
            + [e for e in self.evidence if e.kind != "graph_fact"]
        )
        for item in ordered:
            if item.kind == "graph_fact":
                block = f"{item.citation} DERIVED FROM THE KNOWLEDGE GRAPH\n{item.text}"
            else:
                block = f"{item.citation} {item.title}\n{item.text}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    @property
    def text_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == "text"]

    @property
    def graph_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.kind == "graph_fact"]

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "latency_ms": round(self.latency_ms, 1),
            "text_chunks": len(self.text_evidence),
            "graph_facts": len(self.graph_evidence),
            "entities": len(self.entities),
            "documents": len({e.doc_id for e in self.text_evidence if e.doc_id}),
            "max_hops": max([e.hops for e in self.entities], default=0),
            **self.stats,
        }


# Reciprocal Rank Fusion constant.  60 is the value from the original RRF paper
# and it is not arbitrary: it flattens the contribution of the top ranks so a
# single list cannot dominate the fused order, which is exactly what you want
# when combining a semantic ranking with a lexical one whose scores are on
# completely incomparable scales.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Evidence]], *, limit: int
) -> list[Evidence]:
    """Merge several ranked lists into one.

    Why RRF rather than normalising and adding the scores?  Because cosine
    similarity lives in [0,1] and Lucene's BM25 score is unbounded and
    corpus-dependent.  Min-max normalising them makes the numbers comparable
    only in the sense that they are now both between 0 and 1; it does not make
    them *mean* the same thing, and the weighting you end up with is an
    accident of your corpus.  RRF throws away the scores and uses only rank
    order, which is the part that transfers.
    """
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}
    sources: dict[str, set[str]] = {}

    for source, items in ranked_lists.items():
        for rank, item in enumerate(items):
            scores[item.source_id] = scores.get(item.source_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(item.source_id, set()).add(source)
            # Keep the first-seen copy; they carry the same text either way.
            best.setdefault(item.source_id, item)

    fused: list[Evidence] = []
    for source_id, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = best[source_id]
        found_by = sources[source_id]
        fused.append(
            Evidence(
                kind=item.kind,
                text=item.text,
                source_id=item.source_id,
                doc_id=item.doc_id,
                title=item.title,
                doc_type=item.doc_type,
                score=score,
                retrieved_by="+".join(sorted(found_by)),
                detail=item.detail,
            )
        )
    return fused[:limit]

"""A small retrieval index over a fake refund policy document.

Deliberately not a vector database. A bag-of-words scorer over pre-chunked
passages is enough to demonstrate every retrieval failure the post names, and it
keeps the project runnable with no service to start and no embedding key.

Three things here are load-bearing:

  1. The tenant id is part of the query, not a filter over results. `search`
     never has another tenant's text in memory at any point. See
     tests/test_tenant_scoping.py.

  2. Every chunk carries an effective date and an indexed-at date. The post's
     point is that nothing in a retrieval pipeline can tell you whether the
     passage it found is still true, so the chunk has to say so itself and the
     agent has to be able to quote it.

  3. `corrupt_passage` swaps in a superseded version of the refund window. The
     retrieval still succeeds, the citation is still real, the span is still
     green. That is the failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

TODAY = date(2026, 8, 4)


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    tenant_id: str
    doc: str
    section: str
    text: str
    effective_from: str
    indexed_at: str
    superseded: bool = False

    @property
    def index_age_days(self) -> int:
        y, m, d = (int(p) for p in self.indexed_at.split("-"))
        return (TODAY - date(y, m, d)).days


# The current corpus for tenant-northwind. Chunked by section, which the post
# calls a design decision rather than a default.
CORPUS: list[Passage] = [
    Passage(
        chunk_id="nw-refund-001",
        tenant_id="tenant-northwind",
        doc="Northwind Refund Policy",
        section="Return window",
        text=(
            "Customers may return any item within 30 days of delivery for a full "
            "refund. The 30 day window is measured from the delivery date recorded "
            "by the courier, not the order date."
        ),
        effective_from="2026-07-01",
        indexed_at="2026-07-01",
    ),
    Passage(
        chunk_id="nw-refund-002",
        tenant_id="tenant-northwind",
        doc="Northwind Refund Policy",
        section="Return shipping fee",
        text=(
            "Return shipping is free for gold tier customers. Standard tier "
            "customers are charged a flat 8.00 USD return shipping fee, deducted "
            "from the refund."
        ),
        effective_from="2026-07-01",
        indexed_at="2026-07-01",
    ),
    Passage(
        chunk_id="nw-refund-003",
        tenant_id="tenant-northwind",
        doc="Northwind Refund Policy",
        section="Items in transit",
        text=(
            "An order that has shipped but not yet been delivered cannot be "
            "returned. The customer may refuse delivery, which returns the parcel "
            "to the warehouse and triggers an automatic refund within 5 business days."
        ),
        effective_from="2026-07-01",
        indexed_at="2026-07-01",
    ),
    Passage(
        chunk_id="nw-refund-004",
        tenant_id="tenant-northwind",
        doc="Northwind Refund Policy",
        section="Damaged goods",
        text=(
            "Items that arrive damaged are refunded in full with no return shipping "
            "fee regardless of tier, provided the damage is reported within 14 days "
            "of delivery."
        ),
        effective_from="2026-07-01",
        indexed_at="2026-07-01",
    ),
    # A different retailer, in the same index. This is the row that must never
    # be retrieved by a tenant-northwind query.
    Passage(
        chunk_id="ct-refund-001",
        tenant_id="tenant-contoso",
        doc="Contoso Returns Policy",
        section="Return window",
        text=(
            "Contoso accepts returns within 90 days of purchase. Contoso absorbs "
            "all return shipping costs. Contoso internal margin note: returns above "
            "300 USD are routed to the fraud desk."
        ),
        effective_from="2026-01-01",
        indexed_at="2026-01-01",
    ),
]

# The superseded version of the return window. It was true until 2026-07-01 and
# is still sitting in the index because nobody deleted it on reindex. Retrieval
# has no way to know. This is what the post calls the most dangerous failure.
STALE_REFUND_WINDOW = Passage(
    chunk_id="nw-refund-001",
    tenant_id="tenant-northwind",
    doc="Northwind Refund Policy",
    section="Return window",
    text=(
        "Customers may return any item within 14 days of delivery for a full "
        "refund. The 14 day window is measured from the delivery date recorded "
        "by the courier, not the order date."
    ),
    effective_from="2025-02-01",
    indexed_at="2025-02-01",
    superseded=True,
)

_STOP = {
    "the", "a", "an", "is", "are", "of", "for", "to", "in", "on", "my", "i",
    "can", "do", "does", "it", "and", "was", "how", "what", "if", "be",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _score(query: str, passage: Passage) -> float:
    q = set(_tokens(query))
    if not q:
        return 0.0
    body = _tokens(passage.section + " " + passage.text)
    if not body:
        return 0.0
    overlap = sum(1 for t in body if t in q)
    return overlap / (len(body) ** 0.5)


def search(
    *,
    tenant_id: str,
    query: str,
    top_k: int = 3,
    corrupt: bool = False,
) -> dict[str, Any]:
    """Retrieve passages for one tenant.

    The tenant id is a parameter of the search itself. The candidate list is
    built from rows that already match it, so another tenant's text is never
    loaded and then discarded. Doing it the other way round is a breach whether
    or not the text reaches the answer.

    There is deliberately no parameter that widens this. The cross-tenant
    failure toggle can change the *query text* to ask for another retailer's
    policy, and it still cannot reach it, because the scope is not derived from
    the query.
    """
    scoped_tenant = tenant_id

    # SCOPE BOUNDARY. Everything after this line is already tenant-scoped.
    candidates = [p for p in CORPUS if p.tenant_id == scoped_tenant]

    ranked = sorted(
        ((p, _score(query, p)) for p in candidates), key=lambda x: x[1], reverse=True
    )
    hits = [p for p, s in ranked[:top_k] if s > 0]

    warnings: list[str] = []
    if corrupt:
        # Swap the current return-window chunk for its superseded version. Same
        # chunk id, same document, same section, real citation. Nothing
        # downstream can tell, and no span turns red.
        hits = [STALE_REFUND_WINDOW if p.chunk_id == "nw-refund-001" else p for p in hits]

    max_age = max((p.index_age_days for p in hits), default=0)
    if max_age > 90:
        # Index age is the metric that catches a stale corpus, and it is the one
        # almost nobody has on a dashboard. It is a warning, not an error,
        # because the request genuinely did succeed.
        warnings.append(
            f"index age {max_age}d exceeds the 90d threshold; a passage in this "
            "answer may be superseded"
        )

    return {
        "tenant_id": scoped_tenant,
        "query": query,
        "passages": [
            {
                "chunk_id": p.chunk_id,
                "doc": p.doc,
                "section": p.section,
                "text": p.text,
                "effective_from": p.effective_from,
                "indexed_at": p.indexed_at,
                "index_age_days": p.index_age_days,
            }
            for p in hits
        ],
        "max_index_age_days": max_age,
        "warnings": warnings,
    }

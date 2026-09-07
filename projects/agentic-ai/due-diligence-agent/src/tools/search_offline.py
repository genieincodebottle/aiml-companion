"""Canned search results, so the demo needs no network.

The real `WebSearch` falls back to DuckDuckGo when no Tavily key is present,
which still requires a connection. Offline mode swaps the whole tool out so the
pipeline can run on a train.

The results deliberately include one thin, low-authority hit, because an agent
that only ever sees clean sources never has to decide what to trust.
"""
from __future__ import annotations

from src.tools.search import SearchResult

OFFLINE_SOURCE = "offline_fixture"


class OfflineSearch:
    """Same surface as WebSearch.search, with fixed results."""

    def __init__(self, *_a, **_kw):
        self.queries: list[str] = []

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        self.queries.append(query)
        rows = [
            SearchResult(
                title="Quarterly filing extract",
                url="offline://filings/q3",
                snippet=("Revenue rose 18 percent year on year. Free cash flow fell "
                         "to $310M from $402M."),
                source=OFFLINE_SOURCE,
                relevance_score=0.94,
            ),
            SearchResult(
                title="Guidance withdrawn",
                url="offline://news/1",
                snippet=("Management withdrew full-year guidance, citing limited "
                         "visibility on demand."),
                source=OFFLINE_SOURCE,
                relevance_score=0.91,
            ),
            SearchResult(
                title="Industry blog post",
                url="offline://blog/1",
                snippet="An unsourced claim that a large customer is leaving.",
                source=OFFLINE_SOURCE,
                relevance_score=0.31,
            ),
        ]
        return rows[: (max_results or len(rows))]

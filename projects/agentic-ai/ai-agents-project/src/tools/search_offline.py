"""Deterministic search results, so the pipeline runs without a Tavily key.

The researcher agents are the only thing standing between "I cloned this" and
"I watched it work". Without `TAVILY_API_KEY` every search returned `[]`, the
quality gate reported "no sources to evaluate", and the run finished with an
empty report.

These sources are fabricated and the snippets say so where it matters. They
exist to exercise the pipeline, not to teach anyone about the topic. Two things
are deliberate:

- The results DISAGREE with each other. One reports falling unit costs and
  another reports rising total spend. Without a disagreement the analyst's
  conflict detection and the cross-referencing step never fire, and a reader
  never sees the part of the system that earns its keep.
- One source is low quality, with a thin snippet and a self-published domain,
  so the quality gate has something to actually reject.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


# Domains carry an implied credibility in the quality gate, so the mix matters.
_TEMPLATES = [
    {
        "title": "State of the field: adoption survey {year}",
        "url": "https://example-research.org/survey-{slug}",
        "snippet": (
            "Adoption grew year on year across every segment measured, with the "
            "sharpest increase among teams of under fifty engineers. Reported "
            "production reliability, however, sat well below the figures quoted "
            "in vendor demonstrations."
        ),
        "date": "2026-02-11",
    },
    {
        "title": "Practitioner report: what {slug} costs in practice",
        "url": "https://engineering.example.com/cost-of-{slug}",
        "snippet": (
            "Per-call pricing has fallen repeatedly over the period. Total spend "
            "nonetheless rose for most teams in the sample, because call volume "
            "grew faster than unit prices fell."
        ),
        "date": "2026-01-28",
    },
    {
        "title": "Vendor pricing update: {slug}",
        "url": "https://vendor.example.com/blog/pricing-{slug}",
        "snippet": (
            "Costs continue to fall and will keep falling. No independent "
            "benchmark covers this area yet."
        ),
        "date": "2026-03-02",
    },
    {
        "title": "Notes on {slug}",
        "url": "https://someones-blog.example.net/{slug}",
        "snippet": "Short post. Mostly opinion, no sources cited.",
        "date": "",
    },
    {
        "title": "Limitations and criticism of {slug}",
        "url": "https://example-university.edu/papers/{slug}",
        "snippet": (
            "The authors identify three recurring failure modes and argue that "
            "benchmark results overstate real-world performance because the "
            "evaluation sets omit the long tail."
        ),
        "date": "2025-11-19",
    },
]


def _slug(query: str) -> str:
    words = [w.lower() for w in query.split() if w.isalnum() or w.isalpha()]
    return "-".join(words[:4]) or "topic"


def offline_search(query: str, max_results: int = 5) -> list[dict]:
    """Return stable, query-derived sources in the shape `web_search` returns.

    Deterministic for a given query: the same sub-topic always yields the same
    sources, so two runs of the demo are comparable and the cache has something
    real to hit.
    """
    slug = _slug(query)
    # Rotate the starting template by query, so parallel researchers on
    # different sub-topics do not all return an identical list.
    offset = int(hashlib.sha256(query.encode()).hexdigest(), 16) % len(_TEMPLATES)
    sources = []
    for i in range(min(max_results, len(_TEMPLATES))):
        t = _TEMPLATES[(offset + i) % len(_TEMPLATES)]
        sources.append({
            "title": t["title"].format(slug=slug, year=2026),
            "url": t["url"].format(slug=slug),
            "snippet": t["snippet"],
            "date": t["date"],
            "tool": "offline",
        })
    logger.info(f"Offline search: '{query[:50]}' returned {len(sources)} results")
    return sources

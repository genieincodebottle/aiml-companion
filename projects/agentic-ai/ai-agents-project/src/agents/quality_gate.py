# ============================================
# Quality Gate - Pure Python Source Scoring
# ============================================
# Scores source quality WITHOUT an LLM call (pure Python heuristics).
# Routes the pipeline: quality_passed=True -> analyst, False -> retry research.
# Teaches: conditional routing based on computed metrics, not LLM judgment.

import logging
from src.config import get_pipeline_config

logger = logging.getLogger(__name__)

# Domain trust scores (higher = more trustworthy)
DOMAIN_TRUST = {
    "wikipedia.org": 0.9,
    "arxiv.org": 0.95,
    "github.com": 0.7,
    "nature.com": 0.95,
    "ieee.org": 0.9,
    "acm.org": 0.9,
    "scholar.google.com": 0.85,
    "medium.com": 0.5,
    "reddit.com": 0.3,
    "quora.com": 0.3,
}


def _domain_score(url: str) -> float:
    """Score a source URL based on domain trustworthiness."""
    for domain, score in DOMAIN_TRUST.items():
        if domain in url:
            return score
    return 0.5  # Unknown domains get neutral score


def _snippet_score(snippet: str) -> float:
    """Score a snippet based on content quality heuristics."""
    if not snippet:
        return 0.0

    score = 0.0
    length = len(snippet)

    # Length: prefer substantial snippets
    if length > 200:
        score += 0.3
    elif length > 100:
        score += 0.2
    elif length > 50:
        score += 0.1

    # Contains numbers/data (likely factual)
    import re
    if re.search(r'\d+\.?\d*%|\$\d+|\d{4}', snippet):
        score += 0.2

    # Contains technical terms (likely relevant)
    tech_terms = ["algorithm", "model", "data", "research", "study", "analysis",
                  "performance", "accuracy", "training", "network", "learning"]
    term_count = sum(1 for t in tech_terms if t.lower() in snippet.lower())
    score += min(term_count * 0.05, 0.3)

    # Penalize very short or empty snippets
    if length < 20:
        score -= 0.2

    return min(max(score, 0.0), 1.0)


def quality_gate(state: dict) -> dict:
    """Score source quality and decide whether to proceed or retry.

    Pure Python - no LLM call. Scores each source on domain trust
    and snippet quality, then computes an aggregate quality score.

    Returns {quality_score, quality_passed, source_ranking, pipeline_trace}.
    """
    sources = state.get("sources", [])
    cfg = get_pipeline_config()
    threshold = cfg.get("quality_threshold", 0.4)

    if not sources:
        logger.warning("Quality Gate: no sources to evaluate")
        return {
            "quality_score": 0.0,
            "quality_passed": False,
            "source_ranking": [],
            "pipeline_trace": [{
                "agent": "quality_gate",
                "duration_ms": 0,
                "tokens": 0,
                "summary": "No sources - routing to retry",
            }],
        }

    # Score each source
    scored = []
    for i, src in enumerate(sources):
        d_score = _domain_score(src.get("url", ""))
        s_score = _snippet_score(src.get("snippet", ""))
        combined = 0.6 * d_score + 0.4 * s_score
        scored.append({
            "index": i,
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            "domain_score": round(d_score, 2),
            "snippet_score": round(s_score, 2),
            "combined_score": round(combined, 2),
        })

    # Sort by combined score descending
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    # Aggregate quality = mean of top sources
    top_scores = [s["combined_score"] for s in scored[:5]]
    quality_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    quality_passed = quality_score >= threshold

    logger.info(
        f"Quality Gate: score={quality_score:.2f}, threshold={threshold}, "
        f"passed={quality_passed}, sources={len(sources)}"
    )

    return {
        "quality_score": round(quality_score, 3),
        "quality_passed": quality_passed,
        "source_ranking": scored,
        "pipeline_trace": [{
            "agent": "quality_gate",
            "duration_ms": 0,
            "tokens": 0,
            "summary": f"Score {quality_score:.2f}/{threshold} - {'PASS' if quality_passed else 'RETRY'}",
        }],
    }

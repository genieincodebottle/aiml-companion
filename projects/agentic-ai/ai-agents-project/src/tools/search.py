# ============================================
# Tavily Web Search Tool
# ============================================

import logging
import os

logger = logging.getLogger(__name__)

_search = None


def _get_search():
    """Lazy-initialize TavilySearch (needs TAVILY_API_KEY at runtime, not import)."""
    global _search
    if _search is None:
        from langchain_tavily import TavilySearch
        _search = TavilySearch(max_results=5)
    return _search


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily API.

    Returns list of {title, url, snippet, date, tool} dicts.

    With no TAVILY_API_KEY this used to log an error and return [], every
    researcher came back empty, the quality gate reported "no sources to
    evaluate", and the run finished with an empty report and exit code 0.
    Offline mode returns deterministic stand-in sources instead, so the
    pipeline is watchable without a key.
    """
    from ..llm import has_real_key, is_offline

    # `not os.getenv(...)` was the test here, and the placeholder in .env.example
    # is a non-empty string, so this branch was skipped whenever a .env existed.
    if is_offline() or not has_real_key("TAVILY_API_KEY"):
        from .search_offline import offline_search
        return offline_search(query, max_results)

    try:
        raw = _get_search().invoke(query)
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        sources = [
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:500],
                "date": r.get("date", ""),
                "tool": "tavily",
            }
            for r in results[:max_results]
        ]
        logger.info(f"Tavily search: '{query[:50]}' returned {len(sources)} results")
        return sources
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []

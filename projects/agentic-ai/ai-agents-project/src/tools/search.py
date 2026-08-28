# ============================================
# Tavily Web Search Tool
# ============================================

import logging

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
    """
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

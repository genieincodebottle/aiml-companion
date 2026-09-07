# ============================================
# Wikipedia Search Tool
# ============================================
# Uses the Wikipedia REST API (no API key needed).

import logging
import requests

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/api/rest_v1"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"


def wiki_search(query: str, max_results: int = 3) -> list[dict]:
    """Search Wikipedia for articles matching the query.

    Returns list of {title, url, snippet, date, tool} dicts.
    No API key required - uses public REST API.
    """
    try:
        resp = requests.get(
            WIKI_SEARCH,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1,
            },
            timeout=10,
            headers={"User-Agent": "ResearchBot/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            # Clean HTML tags from snippet
            snippet = item.get("snippet", "")
            snippet = snippet.replace("<span class=\"searchmatch\">", "")
            snippet = snippet.replace("</span>", "")
            snippet = snippet.replace("&quot;", '"')
            snippet = snippet.replace("&amp;", "&")

            results.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": snippet[:500],
                "date": item.get("timestamp", ""),
                "tool": "wikipedia",
            })

        logger.info(f"Wikipedia search: '{query[:50]}' returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return []

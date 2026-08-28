"""Web search tool using Tavily API.

Tavily is a search API built for AI agents. Free tier: 1000 searches/month.
Sign up at https://tavily.com to get an API key.

If no API key is set, falls back to a mock search for local testing.
"""

import json
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError

from agents.config import TAVILY_API_KEY, MAX_RESEARCH_RESULTS

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web and return a list of results.

    Each result has: title, url, content (snippet).

    Returns mock data if TAVILY_API_KEY is not set (for local testing).
    """
    num_results = max_results or MAX_RESEARCH_RESULTS

    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set - returning mock search results")
        return _mock_search(query, num_results)

    return _tavily_search(query, num_results)


def _tavily_search(query: str, max_results: int) -> list[dict]:
    """Call Tavily Search API."""
    payload = json.dumps({
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }).encode("utf-8")

    req = Request(
        TAVILY_SEARCH_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TAVILY_API_KEY}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        logger.info("Tavily search returned %d results for: %s", len(results), query)
        return results

    except (URLError, json.JSONDecodeError) as e:
        logger.error("Tavily search failed: %s", e)
        return _mock_search(query, max_results)


def _mock_search(query: str, max_results: int) -> list[dict]:
    """Return mock search results for local development without API key."""
    mock_results = [
        {
            "title": f"Search Result 1 for: {query}",
            "url": "https://example.com/result-1",
            "content": (
                f"This is a mock search result about '{query}'. "
                "In production, this would contain real web content from Tavily search. "
                "Set TAVILY_API_KEY in your .env file to enable real web search."
            ),
        },
        {
            "title": f"Search Result 2 for: {query}",
            "url": "https://example.com/result-2",
            "content": (
                f"Another mock result about '{query}'. "
                "Tavily offers 1000 free searches per month - sign up at tavily.com."
            ),
        },
        {
            "title": f"Search Result 3 for: {query}",
            "url": "https://example.com/result-3",
            "content": (
                f"Third mock result for '{query}'. "
                "The multi-agent system works with or without web search - "
                "agents will use their training knowledge when search is unavailable."
            ),
        },
    ]
    return mock_results[:max_results]


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r['content']}\n"
        )
    return "\n".join(formatted)

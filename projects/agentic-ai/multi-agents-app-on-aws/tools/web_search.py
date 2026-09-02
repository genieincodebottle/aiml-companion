"""Web search tool using Tavily API.

Tavily is a search API built for AI agents. Free tier: 1000 searches/month.
Sign up at https://tavily.com to get an API key.

If no API key is set, returns clearly-labelled placeholder results so the
pipeline can be exercised offline. Those results are tagged `provenance="mock"`
and every layer that touches them says so -- see the note on `SearchUnavailable`
below for why a silent fallback is worse than an error.
"""

import json
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError

from agents.config import TAVILY_API_KEY, MAX_RESEARCH_RESULTS

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchUnavailable(RuntimeError):
    """Raised when a configured search backend fails.

    This exists because the failure path used to `return _mock_search(...)`. A
    timeout, a revoked key, or a 429 therefore turned a research agent into a
    fiction generator: the placeholder text went into the prompt, the model
    summarised it obediently, and the report came back citing
    https://example.com/result-1 as a source. Nothing downstream could tell that
    apart from a real run -- the agent's `sources` list looked identical.

    Missing key -> placeholders, loudly labelled, is a reasonable offline
    default. Configured-but-broken -> placeholders is not: it converts an
    outage you could have seen into evidence you cannot audit.
    """


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web and return a list of results.

    Each result has: title, url, content (snippet).

    Every result carries `provenance`: "tavily" for real hits, "mock" for the
    offline placeholders returned when no API key is configured. Check it before
    you believe a citation.

    Raises:
        SearchUnavailable: if a key IS configured but the search failed.
    """
    num_results = max_results or MAX_RESEARCH_RESULTS

    if not TAVILY_API_KEY:
        logger.warning(
            "TAVILY_API_KEY not set - returning MOCK search results. Any report "
            "built on these is fiction; set the key for real research."
        )
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
                "provenance": "tavily",
            })

        logger.info("Tavily search returned %d results for: %s", len(results), query)
        return results

    except (URLError, json.JSONDecodeError) as e:
        # Deliberately NOT falling back to _mock_search here. See SearchUnavailable.
        logger.error("Tavily search failed: %s", e)
        raise SearchUnavailable(f"Tavily search failed for {query!r}: {e}") from e


def _mock_search(query: str, max_results: int) -> list[dict]:
    """Placeholder results for local development without an API key.

    Tagged `provenance="mock"` so callers, the prompt builder, and anyone
    reading the agent's output can tell these are not evidence.
    """
    mock_results = [
        {
            "title": f"Search Result 1 for: {query}",
            "url": "https://example.com/result-1",
            "content": (
                f"This is a mock search result about '{query}'. "
                "In production, this would contain real web content from Tavily search. "
                "Set TAVILY_API_KEY in your .env file to enable real web search."
            ),
            "provenance": "mock",
        },
        {
            "title": f"Search Result 2 for: {query}",
            "url": "https://example.com/result-2",
            "content": (
                f"Another mock result about '{query}'. "
                "Tavily offers 1000 free searches per month - sign up at tavily.com."
            ),
            "provenance": "mock",
        },
        {
            "title": f"Search Result 3 for: {query}",
            "url": "https://example.com/result-3",
            "content": (
                f"Third mock result for '{query}'. "
                "The multi-agent system works with or without web search - "
                "agents will use their training knowledge when search is unavailable."
            ),
            "provenance": "mock",
        },
    ]
    return mock_results[:max_results]


def format_search_results(results: list[dict]) -> str:
    """Format search results into a readable string for the LLM.

    If any result is a placeholder, the block is prefixed with a warning. The
    model cannot inspect `provenance` -- it only sees this text -- so the label
    has to be in the text, or the placeholders read exactly like sources.
    """
    if not results:
        return "No search results found."

    formatted = []
    if any(r.get("provenance") == "mock" for r in results):
        formatted.append(
            "WARNING: the results below are PLACEHOLDERS, not real search hits "
            "(no search API key is configured). Do not cite them as sources and "
            "do not present their content as fact. Say plainly that web search "
            "was unavailable.\n"
        )

    for i, r in enumerate(results, 1):
        tag = " [PLACEHOLDER - NOT A REAL SOURCE]" if r.get("provenance") == "mock" else ""
        formatted.append(
            f"[{i}] {r['title']}{tag}\n"
            f"    URL: {r['url']}\n"
            f"    {r['content']}\n"
        )
    return "\n".join(formatted)

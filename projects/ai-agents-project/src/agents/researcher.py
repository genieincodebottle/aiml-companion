# ============================================
# Researcher Agent - Multi-Tool Parallel Research
# ============================================
# Searches for sources using multiple tools (Tavily, Wikipedia, scraper).
# Designed to run in parallel via LangGraph Send() - one instance per sub-topic.
# Teaches: parallel fan-out execution, multi-tool selection.

import time
import logging
from src.tools.search import web_search
from src.tools.wikipedia import wiki_search
from src.tools.tool_selector import select_tools
from src.guardrails import check_budget

logger = logging.getLogger(__name__)


def researcher(state: dict) -> dict:
    """Research a single sub-topic using selected tools.

    This function runs once per sub-topic (dispatched via Send()).
    Results are merged into the main state via operator.add reducer.

    Input state must contain:
      - query: the sub-topic to research
      - token_count: current token usage

    Returns {sources, search_queries_used, token_count, pipeline_trace}.
    """
    start = time.time()
    query = state.get("query", "")

    if not check_budget(state.get("token_count", 0)):
        return {
            "sources": [],
            "search_queries_used": [query],
            "errors": [f"Budget exceeded before researching: {query[:50]}"],
            "pipeline_trace": [{"agent": "researcher", "status": "skipped", "reason": "budget"}],
        }

    try:
        tools = select_tools(query)
        all_sources = []
        queries_used = [query]

        # Run selected tools
        for tool_name in tools:
            if tool_name == "tavily":
                results = web_search(query, max_results=5)
                all_sources.extend(results)
            elif tool_name == "wikipedia":
                results = wiki_search(query, max_results=3)
                all_sources.extend(results)

        # Deduplicate by URL
        seen_urls = set()
        unique_sources = []
        for s in all_sources:
            url = s.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(s)

        duration = int((time.time() - start) * 1000)
        logger.info(
            f"Researcher: '{query[:50]}' - {len(unique_sources)} sources "
            f"from {tools} in {duration}ms"
        )

        return {
            "sources": unique_sources,
            "search_queries_used": queries_used,
            "token_count": 300,
            "pipeline_trace": [{
                "agent": "researcher",
                "duration_ms": duration,
                "tokens": 300,
                "summary": f"Found {len(unique_sources)} sources for '{query[:40]}'",
                "tools_used": tools,
            }],
        }
    except Exception as e:
        logger.error(f"Researcher error for '{query[:50]}': {e}")
        return {
            "sources": [],
            "search_queries_used": [query],
            "errors": [f"Researcher error: {e}"],
            "pipeline_trace": [{
                "agent": "researcher",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": f"Error: {e}",
            }],
        }

# ============================================
# Tool Selector - Query Type to Tool Routing
# ============================================
# Decides which research tool to use based on query characteristics.
# Teaches agent tool selection: agents pick tools, not hardcoded routing.

import logging

logger = logging.getLogger(__name__)

# Keywords that suggest Wikipedia is a good source
WIKI_KEYWORDS = [
    "what is", "define", "definition", "history of", "overview",
    "explain", "introduction to", "basics of", "concept of",
    "who is", "when was", "where is",
]

# Keywords that suggest web search is needed (current events, recent)
WEB_KEYWORDS = [
    "latest", "recent", "2024", "2025", "2026", "new", "current",
    "trending", "update", "breakthrough", "release", "announce",
]


def select_tools(query: str) -> list[str]:
    """Decide which tools to use for a given query.

    Returns list of tool names: ["tavily"], ["wikipedia"], or ["tavily", "wikipedia"].
    Always includes at least "tavily" as the primary tool.
    """
    query_lower = query.lower()

    tools = []

    # Check for web-search indicators (current events)
    has_web_signal = any(kw in query_lower for kw in WEB_KEYWORDS)

    # Check for Wikipedia indicators (factual/definitional)
    has_wiki_signal = any(kw in query_lower for kw in WIKI_KEYWORDS)

    # Always include Tavily as primary
    tools.append("tavily")

    # Add Wikipedia for factual/definitional queries (unless clearly about current events only)
    if has_wiki_signal and not has_web_signal:
        tools.append("wikipedia")
    elif has_wiki_signal and has_web_signal:
        # Both signals - use both tools
        tools.append("wikipedia")

    logger.info(f"Tool selection for '{query[:50]}': {tools}")
    return tools

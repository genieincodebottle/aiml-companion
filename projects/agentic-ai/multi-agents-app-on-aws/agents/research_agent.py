"""Research Agent - Searches the web and summarizes findings.

This agent takes a research query, searches the web using Tavily,
and produces a structured summary of findings with sources.
"""

import logging

from agents.config import call_llm, logger
from tools.web_search import SearchUnavailable, web_search, format_search_results

SYSTEM_PROMPT = """You are an expert research analyst. Your job is to research a given topic
and provide comprehensive, well-sourced findings.

Instructions:
1. You will receive a research query and web search results.
2. Synthesize the information into a clear, structured summary.
3. Always cite your sources with [Source N] references.
4. Focus on facts, data, and recent developments.
5. If search results are insufficient, supplement with your training knowledge
   but clearly mark those parts as "Based on general knowledge."
6. Structure your output with clear sections and bullet points.

Output format:
## Key Findings
- Finding 1 [Source 1]
- Finding 2 [Source 2]

## Details
[Detailed paragraphs with source citations]

## Sources
1. [Title](URL)
2. [Title](URL)
"""


def run(input_data: dict) -> dict:
    """Execute the research agent.

    Args:
        input_data: {"query": "topic to research", "context": "optional context"}

    Returns:
        {"result", "sources", "agent", "search_mode"} where `search_mode` is
        "live" (real hits), "placeholder" (no API key -- the findings are not
        evidence) or "unavailable" (search was configured but failed).

    A failed search returns an error result rather than research built on
    placeholder text. The search tool used to swallow the failure and hand back
    mock hits, so an outage produced a confident report citing example.com.
    """
    query = input_data.get("query", "")
    context = input_data.get("context", "")

    if not query:
        return {"result": "Error: no research query provided", "sources": [],
                "agent": "research", "search_mode": "none"}

    logger.info("Research Agent: searching for '%s'", query)

    # Step 1: Search the web
    try:
        search_results = web_search(query)

        # Step 2: With context (e.g. follow-up research), do a refined search
        if context:
            refined_query = f"{query} {context}"
            search_results.extend(web_search(refined_query, max_results=3))
    except SearchUnavailable as e:
        # Fail visibly. Answering anyway from model memory would look identical
        # to a researched answer, and the caller could not tell them apart.
        logger.error("Research Agent: search unavailable - %s", e)
        return {
            "result": (
                f"Research could not be completed: web search is configured but "
                f"failed ({e}). No sources were retrieved, so no findings are "
                f"reported. Retry once search is available."
            ),
            "sources": [],
            "agent": "research",
            "search_mode": "unavailable",
        }

    placeholders = [r for r in search_results if r.get("provenance") == "mock"]
    search_mode = "placeholder" if placeholders else "live"

    # Step 3: Format search results for the LLM
    formatted_results = format_search_results(search_results)

    # Step 4: Ask the LLM to synthesize the findings
    user_message = (
        f"Research Query: {query}\n\n"
        f"{'Additional Context: ' + context + chr(10) + chr(10) if context else ''}"
        f"Web Search Results:\n{formatted_results}\n\n"
        f"Please synthesize these search results into a comprehensive research summary. "
        f"Cite sources using [Source N] format."
    )

    result = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.2,  # Lower temperature for factual research
    )

    # Extract source URLs for metadata, carrying provenance forward so a
    # caller can distinguish a cited source from a placeholder.
    sources = [
        {"title": r["title"], "url": r["url"],
         "provenance": r.get("provenance", "unknown")}
        for r in search_results
    ]

    if search_mode == "placeholder":
        result = (
            "NOTE: web search was unavailable (no API key), so the findings "
            "below rest on placeholder results and the model's own training "
            "data. They are not researched and the citations are not real.\n\n"
            + result
        )

    logger.info(
        "Research Agent: completed with %d sources (search_mode=%s)",
        len(sources), search_mode,
    )

    return {
        "result": result,
        "sources": sources,
        "agent": "research",
        "search_mode": search_mode,
    }


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Latest trends in AI agents 2026"
    print(f"\nResearching: {query}\n{'=' * 60}")
    output = run({"query": query})
    print(output["result"])
    print(f"\n{'=' * 60}")
    print(f"Sources: {len(output['sources'])}")
    for s in output["sources"]:
        print(f"  - {s['title']}: {s['url']}")

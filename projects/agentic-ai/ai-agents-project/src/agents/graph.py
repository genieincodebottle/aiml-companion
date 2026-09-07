# ============================================
# LangGraph Pipeline - Advanced Graph Wiring
# ============================================
# Wires 8 agents with:
#   1. Send() parallel fan-out for sub-topic research
#   2. Conditional routing based on quality scores
#   3. Iterative refinement loop (writer <-> reviewer)
#
# Graph topology:
#   planner -> researcher(s) [parallel] -> quality_gate
#     -> [conditional: retry or proceed]
#     -> analyst -> synthesizer -> writer -> reviewer
#     -> [conditional: revise or END]

import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import Send

load_dotenv()

from src.models.state import ResearchState, default_state
from src.agents.planner import planner
from src.agents.researcher import researcher
from src.agents.quality_gate import quality_gate
from src.agents.analyst import analyst
from src.agents.synthesizer import synthesizer
from src.agents.writer import writer
from src.agents.reviewer import reviewer
from src.config import get_pipeline_config

logger = logging.getLogger(__name__)


# === Routing Functions ===

def route_to_researchers(state: ResearchState) -> list[Send]:
    """Fan-out: dispatch one researcher per sub-topic using Send().

    Each Send() creates a parallel researcher instance with the sub-topic
    as its query. Results merge back via operator.add reducer on 'sources'.
    """
    sub_topics = state.get("sub_topics", [state["query"]])
    sends = []
    for topic in sub_topics:
        sends.append(Send("researcher", {
            "query": topic,
            "token_count": state.get("token_count", 0),
        }))
    logger.info(f"Fan-out: dispatching {len(sends)} parallel researchers")
    return sends


def route_after_quality(state: ResearchState) -> str:
    """Conditional: if quality passes, proceed to analyst. Otherwise retry once.

    On retry, we route to a retry_researcher node that does a broader search.
    If we've already retried (check search_queries_used), go to analyst anyway.
    """
    if state.get("quality_passed", False):
        return "analyst"

    # Check if we already retried (prevent infinite loop)
    queries_used = state.get("search_queries_used", [])
    if len(queries_used) > len(state.get("sub_topics", [])):
        logger.info("Quality gate: already retried, proceeding to analyst")
        return "analyst"

    logger.info("Quality gate: routing to retry_researcher")
    return "retry_researcher"


def route_after_review(state: ResearchState) -> str:
    """Conditional: if review passes, END. Otherwise loop back to writer.

    Respects max_revisions to prevent infinite loops.
    """
    review = state.get("review", {})
    if review.get("passed", False):
        return END

    cfg = get_pipeline_config()
    max_revisions = cfg.get("max_revisions", 2)
    if state.get("revision_count", 0) >= max_revisions:
        logger.info(f"Max revisions ({max_revisions}) reached, ending")
        return END

    logger.info("Review failed, routing back to writer for revision")
    return "writer"


# === Retry Researcher (broader search on quality gate failure) ===

def retry_researcher(state: ResearchState) -> dict:
    """Re-research with a broader query when quality gate fails.

    Appends " comprehensive overview analysis" to trigger broader results.
    """
    query = state.get("query", "")
    broad_query = f"{query} comprehensive overview analysis"
    logger.info(f"Retry researcher: broadening query to '{broad_query[:60]}'")

    # Reuse the researcher function with the broadened query
    result = researcher({
        "query": broad_query,
        "token_count": state.get("token_count", 0),
    })

    # Update trace to indicate retry
    for trace in result.get("pipeline_trace", []):
        trace["agent"] = "retry_researcher"

    return result


# === Build the Graph ===

def build_graph():
    """Build and compile the 8-agent research pipeline with advanced patterns.

    Returns a compiled LangGraph that can be invoked or streamed.
    """
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("retry_researcher", retry_researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)

    # Entry point
    graph.set_entry_point("planner")

    # Planner -> parallel researchers (Send fan-out)
    graph.add_conditional_edges("planner", route_to_researchers, ["researcher"])

    # Researchers merge -> quality gate
    graph.add_edge("researcher", "quality_gate")

    # Quality gate -> conditional: analyst or retry
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality,
        {"analyst": "analyst", "retry_researcher": "retry_researcher"},
    )

    # Retry researcher -> quality gate (re-evaluate)
    graph.add_edge("retry_researcher", "analyst")

    # Analyst -> Synthesizer -> Writer -> Reviewer
    graph.add_edge("analyst", "synthesizer")
    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer", "reviewer")

    # Reviewer -> conditional: END or back to writer (refinement loop)
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", END: END},
    )

    return graph.compile()


def run_pipeline(query: str) -> dict:
    """Run the full 8-agent research pipeline.

    Returns the final state dict with all accumulated results.
    """
    app = build_graph()
    initial = default_state(query)
    result = app.invoke(initial)
    return result


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What are the latest trends in AI agents for 2025?"
    print(f"Researching: {query}\n")

    result = run_pipeline(query)

    print(f"Sub-topics: {result.get('sub_topics', [])}")
    print(f"Sources found: {len(result.get('sources', []))}")
    print(f"Quality score: {result.get('quality_score', 0):.2f}")
    print(f"Claims extracted: {len(result.get('key_claims', []))}")
    print(f"Conflicts: {len(result.get('conflicts', []))}")
    print(f"Drafts: {len(result.get('drafts', []))}")
    print(f"Review score: {result.get('review', {}).get('score', 'N/A')}")
    print(f"Report length: {len(result.get('final_report', ''))} chars")
    print(f"Total tokens: {result.get('token_count', 0)}")

    trace = result.get("pipeline_trace", [])
    print(f"\nPipeline trace ({len(trace)} steps):")
    for step in trace:
        print(f"  {step.get('agent', '?'):20s} | {step.get('duration_ms', 0):5d}ms | "
              f"{step.get('tokens', 0):5d} tokens | {step.get('summary', '')}")

    errors = result.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

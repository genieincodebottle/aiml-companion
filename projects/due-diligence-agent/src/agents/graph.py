"""
LangGraph orchestration - the core pipeline that wires all agents together.

Pipeline flow:
    plan_research -> [financial, news, competitive, risk] (parallel) -> fact_checker -> resolve_contradictions -> synthesize_report

Key features:
- Parallel execution of 4 specialist agents via Send()
- Fact-checking second pass
- Contradiction debate via Lead Analyst
- Self-correction loop (fact_checker -> re-research if enabled)
- Guardrail checks at every node
- Graceful degradation if any agent fails

Usage:
    from src.agents.graph import build_graph, run_pipeline

    result = run_pipeline("Tesla")
    print(result["final_report"])
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import StateGraph, END, Send

from src.config import get_pipeline_config
from src.models.state import DueDiligenceState, default_state
from src.agents.lead_analyst import plan_research, resolve_contradictions, synthesize_report
from src.agents.financial_analyst import financial_analyst
from src.agents.news_sentiment import news_sentiment
from src.agents.competitive_intel import competitive_intel
from src.agents.risk_assessor import risk_assessor
from src.agents.fact_checker import fact_checker
from src.guardrails.manager import GuardrailManager
from src.llm import token_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail-Wrapped Agent Nodes
# ---------------------------------------------------------------------------

def _create_guarded_node(agent_fn, agent_name: str, guardrail_mgr: GuardrailManager):
    """Wrap an agent function with pre/post guardrail checks."""

    def guarded_node(state: DueDiligenceState) -> dict:
        # Pre-check
        pre = guardrail_mgr.pre_check(agent_name)
        if not pre["passed"]:
            logger.warning("[Guard] Blocked %s: %s", agent_name, pre["reason"])
            return {
                "errors": [f"[{agent_name}] Blocked by guardrail: {pre['reason']}"],
                "warnings": [f"{agent_name} skipped due to guardrail: {pre['reason']}"],
                "pipeline_trace": [{
                    "agent": agent_name,
                    "action": "guardrail_block",
                    "summary": pre["reason"],
                    "duration_seconds": 0.0,
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                    "error": pre["reason"],
                    "retry_count": 0,
                }],
            }

        # Execute agent
        result = agent_fn(state)

        # Post-check
        post = guardrail_mgr.post_check(
            agent_name,
            str(result),
            tokens_used=0,
            cost_usd=0.0,
        )
        if post.get("warnings"):
            existing_warnings = result.get("warnings", [])
            result["warnings"] = existing_warnings + post["warnings"]

        return result

    guarded_node.__name__ = f"guarded_{agent_name}"
    return guarded_node


# ---------------------------------------------------------------------------
# Router Nodes
# ---------------------------------------------------------------------------

def _fan_out_to_specialists(state: DueDiligenceState) -> list[Send]:
    """Route to all 4 specialist agents in parallel via LangGraph Send()."""
    return [
        Send("financial_analyst", state),
        Send("news_sentiment", state),
        Send("competitive_intel", state),
        Send("risk_assessor", state),
    ]


def _route_after_fact_check(state: DueDiligenceState) -> str:
    """After fact-checking, decide whether to debate contradictions or go to synthesis."""
    pipeline_config = get_pipeline_config()
    contradictions = state.get("contradictions", [])
    enable_debate = pipeline_config.get("enable_debate", True)

    unresolved = [c for c in contradictions if c.get("status") != "resolved"]

    if unresolved and enable_debate:
        logger.info("Found %d unresolved contradictions, routing to debate", len(unresolved))
        return "resolve_contradictions"
    else:
        logger.info("No contradictions (or debate disabled), routing to synthesis")
        return "synthesize_report"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph(guardrail_mgr: GuardrailManager | None = None) -> Any:
    """Build the LangGraph pipeline.

    Args:
        guardrail_mgr: Optional guardrail manager. Creates default if None.

    Returns:
        Compiled LangGraph application.
    """
    if guardrail_mgr is None:
        guardrail_mgr = GuardrailManager()

    # Create guarded agent nodes
    guarded_plan = _create_guarded_node(plan_research, "lead_analyst_plan", guardrail_mgr)
    guarded_financial = _create_guarded_node(financial_analyst, "financial_analyst", guardrail_mgr)
    guarded_news = _create_guarded_node(news_sentiment, "news_sentiment", guardrail_mgr)
    guarded_competitive = _create_guarded_node(competitive_intel, "competitive_intel", guardrail_mgr)
    guarded_risk = _create_guarded_node(risk_assessor, "risk_assessor", guardrail_mgr)
    guarded_fact_check = _create_guarded_node(fact_checker, "fact_checker", guardrail_mgr)
    guarded_debate = _create_guarded_node(resolve_contradictions, "lead_analyst_debate", guardrail_mgr)
    guarded_synthesis = _create_guarded_node(synthesize_report, "lead_analyst_synthesis", guardrail_mgr)

    # Build state graph
    graph = StateGraph(DueDiligenceState)

    # Add nodes
    graph.add_node("plan_research", guarded_plan)
    graph.add_node("financial_analyst", guarded_financial)
    graph.add_node("news_sentiment", guarded_news)
    graph.add_node("competitive_intel", guarded_competitive)
    graph.add_node("risk_assessor", guarded_risk)
    graph.add_node("fact_checker", guarded_fact_check)
    graph.add_node("resolve_contradictions", guarded_debate)
    graph.add_node("synthesize_report", guarded_synthesis)

    # Define edges
    # 1. Start -> Planning
    graph.set_entry_point("plan_research")

    # 2. Planning -> Parallel specialists (via conditional edges with Send)
    graph.add_conditional_edges(
        "plan_research",
        _fan_out_to_specialists,
        ["financial_analyst", "news_sentiment", "competitive_intel", "risk_assessor"],
    )

    # 3. All specialists -> Fact checker (converge)
    graph.add_edge("financial_analyst", "fact_checker")
    graph.add_edge("news_sentiment", "fact_checker")
    graph.add_edge("competitive_intel", "fact_checker")
    graph.add_edge("risk_assessor", "fact_checker")

    # 4. Fact checker -> Debate or Synthesis (conditional)
    graph.add_conditional_edges(
        "fact_checker",
        _route_after_fact_check,
        {
            "resolve_contradictions": "resolve_contradictions",
            "synthesize_report": "synthesize_report",
        },
    )

    # 5. Debate -> Synthesis
    graph.add_edge("resolve_contradictions", "synthesize_report")

    # 6. Synthesis -> END
    graph.add_edge("synthesize_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    company_name: str,
    query: str = "",
    depth: str = "standard",
    guardrail_mgr: GuardrailManager | None = None,
    stream: bool = False,
) -> dict:
    """Run the full due diligence pipeline.

    Args:
        company_name: Target company/entity name.
        query: Optional specific research query.
        depth: Analysis depth - "quick", "standard", or "deep".
        guardrail_mgr: Optional guardrail manager.
        stream: If True, yield intermediate states.

    Returns:
        Final pipeline state dict with report, findings, and metadata.
    """
    if guardrail_mgr is None:
        guardrail_mgr = GuardrailManager()

    guardrail_mgr.start_pipeline()

    app = build_graph(guardrail_mgr)
    initial_state = default_state(company_name, query, depth)

    logger.info("Starting due diligence pipeline for: %s (depth: %s)", company_name, depth)
    start_time = time.time()

    if stream:
        # Return generator for streaming updates
        return _stream_pipeline(app, initial_state, guardrail_mgr, start_time)

    # Non-streaming: run to completion
    try:
        final_state = app.invoke(initial_state)

        # Attach metadata
        total_duration = time.time() - start_time
        final_state["total_tokens_used"] = token_tracker.total_tokens
        final_state["total_cost_usd"] = token_tracker.cost_estimate()
        final_state["_execution_time_seconds"] = round(total_duration, 2)
        final_state["_budget_status"] = guardrail_mgr.get_budget_status()
        final_state["_token_summary"] = token_tracker.summary()

        logger.info(
            "Pipeline complete: %s | Risk: %s | Confidence: %.0f%% | Duration: %.1fs",
            company_name,
            final_state.get("overall_risk_rating", "unknown"),
            final_state.get("overall_confidence", 0) * 100,
            total_duration,
        )

        return final_state

    except Exception as e:
        logger.error("Pipeline failed for %s: %s", company_name, e)
        elapsed = time.time() - start_time
        return {
            **initial_state,
            "status": "failed",
            "errors": [f"Pipeline failed: {e}"],
            "final_report": f"# Due Diligence Report: {company_name}\n\n"
                           f"**PIPELINE FAILED:** {e}\n\n"
                           f"Partial findings may be available in the individual sections below.",
            "_execution_time_seconds": round(elapsed, 2),
        }


def _stream_pipeline(app, initial_state, guardrail_mgr, start_time):
    """Generator that yields intermediate states for streaming UI."""
    try:
        for step in app.stream(initial_state, stream_mode="updates"):
            yield {
                "type": "update",
                "data": step,
                "budget": guardrail_mgr.get_budget_status(),
                "elapsed": round(time.time() - start_time, 2),
            }

        yield {
            "type": "complete",
            "budget": guardrail_mgr.get_budget_status(),
            "elapsed": round(time.time() - start_time, 2),
            "tokens": token_tracker.summary(),
        }

    except Exception as e:
        yield {
            "type": "error",
            "error": str(e),
            "elapsed": round(time.time() - start_time, 2),
        }

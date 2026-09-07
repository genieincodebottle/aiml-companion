"""
Financial Analyst Agent - Researches financial health, ratios, and trends.

Responsible for:
- Revenue and profitability analysis
- Financial ratio calculation
- Cash flow assessment
- Identifying financial red/green flags
- Citing data sources for every claim
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import get_agent_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import FinancialAnalysis
from src.models.state import AgentFinding, AgentTrace, DueDiligenceState
from src.tools.search import WebSearchTool

logger = logging.getLogger(__name__)

AGENT_NAME = "financial_analyst"

SYSTEM_PROMPT = """You are a Financial Analyst performing due diligence research.

Your job is to find and analyze financial data about the target company.

RULES:
1. Only state facts you can verify from search results. Never fabricate numbers.
2. If financial data is unavailable (private company, limited data), explicitly say so.
3. Cite the source URL for every financial figure you mention.
4. Distinguish between confirmed data and estimates/projections.
5. Flag data gaps honestly - "insufficient data" is better than a guess.

Focus areas:
- Revenue trends (growth rate, consistency)
- Profitability (margins, net income trajectory)
- Balance sheet health (debt levels, liquidity)
- Cash flow patterns
- Any recent financial events (fundraising, acquisitions, write-downs)

Output your analysis as structured data following the schema exactly."""


def financial_analyst(state: DueDiligenceState) -> dict:
    """Financial Analyst agent node for LangGraph.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with financial findings and trace.
    """
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    plan = state.get("research_plan", {})
    focus = plan.get("financial_analyst", {}).get("focus", "financial health and performance")

    logger.info("[%s] Starting analysis for: %s", AGENT_NAME, company)

    try:
        # Step 1: Search for financial data
        search = WebSearchTool()
        queries = [
            f"{company} financial performance revenue profit 2024 2025",
            f"{company} funding valuation balance sheet",
            f"{company} financial news quarterly results",
        ]

        all_results = []
        for q in queries:
            results = search.search(q, max_results=5)
            all_results.extend(results)

        if not all_results:
            logger.warning("[%s] No search results found for %s", AGENT_NAME, company)
            return _fallback_output(company, "No financial data found via web search", start_time)

        # Step 2: Build context from search results
        context = _build_search_context(all_results)

        # Step 3: LLM analysis with structured output
        structured_llm = get_structured_llm(FinancialAnalysis, temperature=0.1)

        prompt = f"""{SYSTEM_PROMPT}

TARGET COMPANY: {company}
SPECIFIC FOCUS: {focus}

SEARCH RESULTS:
{context}

Analyze the financial data found above. Be precise about numbers and always cite sources.
If data is limited or the company is private, state that clearly."""

        analysis: FinancialAnalysis = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Step 4: Convert to findings
        findings = _analysis_to_findings(analysis)
        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="financial_analysis",
            summary=f"Analyzed {company}: {analysis.financial_health_rating} health, "
                    f"{len(analysis.red_flags)} red flags, {len(analysis.green_flags)} green flags",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Completed: %s (%d findings, %.1fs)", AGENT_NAME, analysis.financial_health_rating, len(findings), duration)

        return {
            "financial_findings": findings,
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Failed: %s", AGENT_NAME, e)
        return _fallback_output(company, str(e), start_time)


def _build_search_context(results: list) -> str:
    """Build a formatted context string from search results."""
    lines = []
    seen_urls = set()
    for r in results:
        if r.url in seen_urls:
            continue
        seen_urls.add(r.url)
        lines.append(f"Source: {r.url}\nTitle: {r.title}\nContent: {r.snippet}\n")
    return "\n---\n".join(lines[:15])  # Cap at 15 unique results


def _analysis_to_findings(analysis: FinancialAnalysis) -> list[AgentFinding]:
    """Convert structured analysis to list of AgentFinding dicts."""
    findings = []

    # Key metrics as findings
    for metric in analysis.key_metrics:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="financial",
            title=f"{metric.metric_name}: {metric.value}",
            detail=f"Trend: {metric.trend}. Assessment: {metric.assessment}",
            severity="info" if metric.assessment in ("positive", "neutral") else "medium",
            confidence=0.8 if metric.source else 0.5,
            sources=[metric.source] if metric.source else [],
            verified=False,
            verification_notes="",
        ))

    # Red flags
    for flag in analysis.red_flags:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="financial",
            title=f"Red Flag: {flag}",
            detail=flag,
            severity="high",
            confidence=0.7,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    # Green flags
    for flag in analysis.green_flags:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="financial",
            title=f"Strength: {flag}",
            detail=flag,
            severity="info",
            confidence=0.7,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    # Overall health finding
    findings.append(AgentFinding(
        agent=AGENT_NAME,
        category="financial",
        title=f"Overall Financial Health: {analysis.financial_health_rating}",
        detail=f"Revenue: {analysis.revenue_analysis}\nProfitability: {analysis.profitability_analysis}",
        severity="info" if analysis.financial_health_rating in ("strong", "moderate") else "high",
        confidence=0.75,
        sources=analysis.sources,
        verified=False,
        verification_notes="",
    ))

    # Data gaps
    for gap in analysis.data_gaps:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="financial",
            title=f"Data Gap: {gap}",
            detail=f"Missing information that limits analysis: {gap}",
            severity="low",
            confidence=1.0,
            sources=[],
            verified=True,
            verification_notes="Acknowledged data limitation",
        ))

    return findings


def _fallback_output(company: str, error: str, start_time: float) -> dict:
    """Graceful degradation when the agent fails."""
    duration = time.time() - start_time
    return {
        "financial_findings": [
            AgentFinding(
                agent=AGENT_NAME,
                category="financial",
                title=f"Financial analysis incomplete for {company}",
                detail=f"Could not complete financial analysis. Reason: {error}. "
                       "This does not necessarily indicate a problem with the company.",
                severity="low",
                confidence=0.0,
                sources=[],
                verified=False,
                verification_notes="Agent failed - manual review recommended",
            )
        ],
        "pipeline_trace": [
            AgentTrace(
                agent=AGENT_NAME,
                action="financial_analysis",
                summary=f"FALLBACK: {error}",
                duration_seconds=round(duration, 2),
                tokens_used=0,
                cost_usd=0.0,
                error=error,
                retry_count=0,
            )
        ],
        "errors": [f"[{AGENT_NAME}] {error}"],
    }

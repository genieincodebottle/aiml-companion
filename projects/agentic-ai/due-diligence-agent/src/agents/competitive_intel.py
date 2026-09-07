"""
Competitive Intelligence Agent - Maps market landscape and positioning.

Responsible for:
- Competitor identification and profiling
- Market position assessment
- Competitive advantages (moats)
- Competitive threats
- Industry trends
"""

from __future__ import annotations

import logging
import time

from src.config import get_agent_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import CompetitiveAnalysis
from src.models.state import AgentFinding, AgentTrace, DueDiligenceState
from src.tools.search import WebSearchTool

logger = logging.getLogger(__name__)

AGENT_NAME = "competitive_intel"

SYSTEM_PROMPT = """You are a Competitive Intelligence Analyst performing due diligence research.

Your job is to map the competitive landscape around the target company.

RULES:
1. Identify the company's industry and market segment precisely.
2. List actual competitors with evidence from search results, not guesses.
3. Assess competitive advantages based on observable evidence (patents, market share data, product features).
4. Distinguish between current competitive position and future projections.
5. Cite sources for all competitive claims.
6. If the competitive landscape is unclear, say so rather than speculate.

Focus areas:
- Industry and market segment identification
- Top 3-5 competitors with their strengths/weaknesses
- Company's competitive moats (technology, brand, network effects, cost advantages)
- Competitive threats and market risks
- Industry trends that could impact positioning

Output your analysis as structured data following the schema exactly."""


def competitive_intel(state: DueDiligenceState) -> dict:
    """Competitive Intelligence agent node for LangGraph."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    plan = state.get("research_plan", {})
    focus = plan.get("competitive_intel", {}).get("focus", "competitive landscape and market position")

    logger.info("[%s] Starting analysis for: %s", AGENT_NAME, company)

    try:
        search = WebSearchTool()
        queries = [
            f"{company} competitors market share industry",
            f"{company} vs competitors comparison advantages",
            f"{company} industry trends market analysis 2024 2025",
        ]

        all_results = []
        for q in queries:
            results = search.search(q, max_results=5)
            all_results.extend(results)

        if not all_results:
            return _fallback_output(company, "No competitive data found", start_time)

        context = _build_search_context(all_results)

        structured_llm = get_structured_llm(CompetitiveAnalysis, temperature=0.2)

        prompt = f"""{SYSTEM_PROMPT}

TARGET COMPANY: {company}
SPECIFIC FOCUS: {focus}

SEARCH RESULTS:
{context}

Analyze the competitive landscape. Identify real competitors from the data and assess market position."""

        analysis: CompetitiveAnalysis = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        findings = _analysis_to_findings(analysis)
        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="competitive_analysis",
            summary=f"Industry: {analysis.industry}, Position: {analysis.market_position}, "
                    f"{len(analysis.competitors)} competitors identified",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Completed: %d competitors, position=%s (%.1fs)", AGENT_NAME, len(analysis.competitors), analysis.market_position, duration)

        return {
            "competitive_findings": findings,
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Failed: %s", AGENT_NAME, e)
        return _fallback_output(company, str(e), start_time)


def _build_search_context(results: list) -> str:
    lines = []
    seen_urls = set()
    for r in results:
        if r.url in seen_urls:
            continue
        seen_urls.add(r.url)
        lines.append(f"Source: {r.url}\nTitle: {r.title}\nContent: {r.snippet}\n")
    return "\n---\n".join(lines[:15])


def _analysis_to_findings(analysis: CompetitiveAnalysis) -> list[AgentFinding]:
    findings = []

    # Market position
    findings.append(AgentFinding(
        agent=AGENT_NAME,
        category="competitive",
        title=f"Market Position: {analysis.market_position} in {analysis.industry}",
        detail=analysis.differentiation_summary,
        severity="info",
        confidence=0.7,
        sources=analysis.sources,
        verified=False,
        verification_notes="",
    ))

    # Competitors
    for comp in analysis.competitors:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="competitive",
            title=f"Competitor: {comp.name} ({comp.market_position})",
            detail=f"Strengths: {', '.join(comp.key_strengths[:3])}\n"
                   f"Weaknesses: {', '.join(comp.key_weaknesses[:3])}\n"
                   f"Market share: {comp.estimated_market_share}",
            severity="info",
            confidence=0.65,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    # Competitive advantages
    for adv in analysis.competitive_advantages:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="competitive",
            title=f"Competitive Advantage: {adv}",
            detail=adv,
            severity="info",
            confidence=0.65,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    # Competitive risks
    for risk in analysis.competitive_risks:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="competitive",
            title=f"Competitive Risk: {risk}",
            detail=risk,
            severity="medium",
            confidence=0.6,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    # Market trends
    for trend in analysis.market_trends:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="competitive",
            title=f"Market Trend: {trend}",
            detail=trend,
            severity="info",
            confidence=0.6,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    return findings


def _fallback_output(company: str, error: str, start_time: float) -> dict:
    duration = time.time() - start_time
    return {
        "competitive_findings": [
            AgentFinding(
                agent=AGENT_NAME,
                category="competitive",
                title=f"Competitive analysis incomplete for {company}",
                detail=f"Could not complete competitive analysis. Reason: {error}",
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
                action="competitive_analysis",
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

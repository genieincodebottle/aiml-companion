"""
News & Sentiment Analyst Agent - Scans recent news and public perception.

Responsible for:
- Recent news event timeline
- Sentiment trend analysis
- Media coverage assessment
- Social media signals
- Identifying PR patterns and concerns
"""

from __future__ import annotations

import logging
import time

from src.config import get_agent_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import NewsSentimentAnalysis
from src.models.state import AgentFinding, AgentTrace, DueDiligenceState
from src.tools.search import WebSearchTool
from src.tools.calculators import calculate_sentiment_score

logger = logging.getLogger(__name__)

AGENT_NAME = "news_sentiment"

SYSTEM_PROMPT = """You are a News & Sentiment Analyst performing due diligence research.

Your job is to scan recent news and assess public perception of the target company.

RULES:
1. Focus on the last 12-18 months of news unless older events are highly significant.
2. Distinguish between factual reporting and opinion pieces.
3. Cite the source URL for every news event.
4. Look for patterns: is coverage increasing/decreasing? Is sentiment shifting?
5. Note the absence of news too - a company with no recent coverage is notable.
6. Do NOT fabricate events. Only report what you find in search results.

Focus areas:
- Major news events (funding, layoffs, product launches, partnerships, lawsuits)
- Media sentiment trend (improving, declining, stable)
- Public perception and brand reputation
- Any crisis events or controversy
- Social media signals if available

Output your analysis as structured data following the schema exactly."""


def news_sentiment(state: DueDiligenceState) -> dict:
    """News & Sentiment agent node for LangGraph."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    plan = state.get("research_plan", {})
    focus = plan.get("news_sentiment", {}).get("focus", "recent news and public perception")

    logger.info("[%s] Starting analysis for: %s", AGENT_NAME, company)

    try:
        search = WebSearchTool()
        queries = [
            f"{company} latest news 2024 2025",
            f"{company} controversy lawsuit regulatory issues",
            f"{company} reviews reputation employee sentiment",
        ]

        all_results = []
        for q in queries:
            results = search.search(q, max_results=5)
            all_results.extend(results)

        if not all_results:
            return _fallback_output(company, "No news articles found", start_time)

        context = _build_search_context(all_results)

        structured_llm = get_structured_llm(NewsSentimentAnalysis, temperature=0.2)

        prompt = f"""{SYSTEM_PROMPT}

TARGET COMPANY: {company}
SPECIFIC FOCUS: {focus}

SEARCH RESULTS:
{context}

Analyze the news and sentiment data found above. Create a timeline of key events and assess overall sentiment."""

        analysis: NewsSentimentAnalysis = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Calculate quantitative sentiment score
        events_for_calc = [
            {"sentiment": e.sentiment, "impact": e.impact}
            for e in analysis.key_events
        ]
        sentiment_score = calculate_sentiment_score(events_for_calc)

        findings = _analysis_to_findings(analysis, sentiment_score)
        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="news_sentiment_analysis",
            summary=f"Found {len(analysis.key_events)} events, "
                    f"sentiment: {analysis.overall_sentiment} ({analysis.sentiment_trend}), "
                    f"score: {sentiment_score['score']}",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Completed: %s sentiment, %d events (%.1fs)", AGENT_NAME, analysis.overall_sentiment, len(analysis.key_events), duration)

        return {
            "news_findings": findings,
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


def _analysis_to_findings(analysis: NewsSentimentAnalysis, sentiment_score: dict) -> list[AgentFinding]:
    findings = []

    # Overall sentiment finding
    findings.append(AgentFinding(
        agent=AGENT_NAME,
        category="news",
        title=f"Overall Sentiment: {analysis.overall_sentiment} (trend: {analysis.sentiment_trend})",
        detail=f"Public perception: {analysis.public_perception}\n"
               f"Media coverage: {analysis.media_coverage_volume}\n"
               f"Quantitative score: {sentiment_score['score']} ({sentiment_score['label']})",
        severity="info" if sentiment_score["score"] > -0.15 else "medium" if sentiment_score["score"] > -0.5 else "high",
        confidence=0.7,
        sources=analysis.sources,
        verified=False,
        verification_notes="",
    ))

    # Key events
    for event in analysis.key_events:
        severity = "info"
        if event.sentiment == "negative" and event.impact == "high":
            severity = "high"
        elif event.sentiment == "negative":
            severity = "medium"

        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="news",
            title=f"[{event.date}] {event.headline}",
            detail=f"{event.summary}\nSentiment: {event.sentiment}, Impact: {event.impact}",
            severity=severity,
            confidence=0.8,
            sources=[event.source],
            verified=False,
            verification_notes="",
        ))

    # Concerns
    for concern in analysis.potential_concerns:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="news",
            title=f"Concern: {concern}",
            detail=concern,
            severity="medium",
            confidence=0.6,
            sources=analysis.sources,
            verified=False,
            verification_notes="",
        ))

    return findings


def _fallback_output(company: str, error: str, start_time: float) -> dict:
    duration = time.time() - start_time
    return {
        "news_findings": [
            AgentFinding(
                agent=AGENT_NAME,
                category="news",
                title=f"News analysis incomplete for {company}",
                detail=f"Could not complete news analysis. Reason: {error}",
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
                action="news_sentiment_analysis",
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

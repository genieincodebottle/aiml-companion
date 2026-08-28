"""
Risk Assessment Agent - Identifies and scores multi-dimensional risks.

Responsible for:
- Legal and regulatory risks
- Operational risks
- Reputational risks
- Technology risks
- ESG concerns
- Risk severity and likelihood scoring
"""

from __future__ import annotations

import logging
import time

from src.config import get_agent_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import RiskAssessment
from src.models.state import AgentFinding, AgentTrace, DueDiligenceState
from src.tools.search import WebSearchTool
from src.tools.calculators import calculate_risk_score

logger = logging.getLogger(__name__)

AGENT_NAME = "risk_assessor"

SYSTEM_PROMPT = """You are a Risk Assessment Specialist performing due diligence research.

Your job is to identify and categorize risks associated with the target company.

RULES:
1. Categorize each risk by type: legal, regulatory, operational, reputational, financial, strategic, technology.
2. Rate severity (critical/high/medium/low) and likelihood (very_likely/likely/possible/unlikely/rare).
3. Only flag risks supported by evidence from search results.
4. Distinguish between confirmed risks and potential risks.
5. Include known mitigations or remediation steps if available.
6. "No significant risks found" is a valid finding if backed by evidence.

Focus areas:
- Legal issues (lawsuits, IP disputes, regulatory actions)
- Regulatory environment (compliance requirements, pending regulations)
- Operational risks (supply chain, key person dependency, technology debt)
- Reputational risks (public controversies, employee reviews)
- ESG concerns (environmental impact, governance structure, social issues)

Output your analysis as structured data following the schema exactly."""


def risk_assessor(state: DueDiligenceState) -> dict:
    """Risk Assessment agent node for LangGraph."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    plan = state.get("research_plan", {})
    focus = plan.get("risk_assessor", {}).get("focus", "legal, regulatory, and operational risks")

    logger.info("[%s] Starting analysis for: %s", AGENT_NAME, company)

    try:
        search = WebSearchTool()
        queries = [
            f"{company} lawsuit legal issues regulatory action",
            f"{company} risks controversy problems complaints",
            f"{company} ESG environmental governance employee reviews",
        ]

        all_results = []
        for q in queries:
            results = search.search(q, max_results=5)
            all_results.extend(results)

        if not all_results:
            return _fallback_output(company, "No risk-related data found", start_time)

        context = _build_search_context(all_results)

        structured_llm = get_structured_llm(RiskAssessment, temperature=0.1)

        prompt = f"""{SYSTEM_PROMPT}

TARGET COMPANY: {company}
SPECIFIC FOCUS: {focus}

SEARCH RESULTS:
{context}

Identify and rate all risks found in the data. Be thorough but evidence-based."""

        analysis: RiskAssessment = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Calculate quantitative risk score
        risks_for_calc = [
            {"severity": r.severity, "likelihood": r.likelihood}
            for r in analysis.risks
        ]
        risk_score = calculate_risk_score(risks_for_calc)

        findings = _analysis_to_findings(analysis, risk_score)
        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="risk_assessment",
            summary=f"Risk level: {analysis.overall_risk_level}, "
                    f"{len(analysis.risks)} risks identified, "
                    f"score: {risk_score['score']} ({risk_score['rating']})",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Completed: %s risk level, %d risks (%.1fs)", AGENT_NAME, analysis.overall_risk_level, len(analysis.risks), duration)

        return {
            "risk_findings": findings,
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


def _analysis_to_findings(analysis: RiskAssessment, risk_score: dict) -> list[AgentFinding]:
    findings = []

    # Overall risk level
    findings.append(AgentFinding(
        agent=AGENT_NAME,
        category="risk",
        title=f"Overall Risk: {analysis.overall_risk_level} (score: {risk_score['score']})",
        detail=f"{analysis.risk_summary}\n\n"
               f"Regulatory: {analysis.regulatory_environment}\n"
               f"Legal: {analysis.legal_history or 'No significant legal history found'}\n"
               f"ESG: {analysis.esg_concerns or 'No significant ESG concerns found'}",
        severity="high" if analysis.overall_risk_level in ("critical", "high") else "medium" if analysis.overall_risk_level == "moderate" else "info",
        confidence=0.7,
        sources=analysis.sources,
        verified=False,
        verification_notes="",
    ))

    # Individual risks
    for risk in analysis.risks:
        findings.append(AgentFinding(
            agent=AGENT_NAME,
            category="risk",
            title=f"[{risk.risk_type.upper()}] {risk.title}",
            detail=f"{risk.description}\n"
                   f"Severity: {risk.severity}, Likelihood: {risk.likelihood}\n"
                   f"Mitigation: {risk.mitigation or 'None identified'}",
            severity=risk.severity,
            confidence=0.65,
            sources=[risk.source] if risk.source else analysis.sources,
            verified=False,
            verification_notes="",
        ))

    return findings


def _fallback_output(company: str, error: str, start_time: float) -> dict:
    duration = time.time() - start_time
    return {
        "risk_findings": [
            AgentFinding(
                agent=AGENT_NAME,
                category="risk",
                title=f"Risk assessment incomplete for {company}",
                detail=f"Could not complete risk assessment. Reason: {error}",
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
                action="risk_assessment",
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

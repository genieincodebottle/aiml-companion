"""
Fact Checker Agent - Cross-references claims and flags contradictions.

Runs as a SECOND PASS after all specialist agents complete.
This is the critical trust layer that challenges other agents' findings.

Responsible for:
- Verifying key claims with independent searches
- Detecting contradictions between agents
- Assigning verification status and confidence
- Flagging unverifiable claims
"""

from __future__ import annotations

import logging
import time

from src.config import get_agent_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import FactCheckReport, ClaimVerification
from src.models.state import AgentFinding, AgentTrace, DueDiligenceState
from src.tools.search import WebSearchTool

logger = logging.getLogger(__name__)

AGENT_NAME = "fact_checker"

SYSTEM_PROMPT = """You are a Fact Verification Specialist performing due diligence quality control.

Your job is to verify claims made by other research agents. You receive their findings
and must independently check the most important ones.

RULES:
1. Focus verification on HIGH-IMPACT claims (financial figures, risk assessments, major events).
2. For each claim, search independently and compare what you find vs what was claimed.
3. Assign verification status:
   - confirmed: Independent evidence supports the claim
   - partially_confirmed: Evidence supports part of the claim but not all details
   - unverifiable: Cannot find independent evidence for or against
   - contradicted: Independent evidence contradicts the claim
   - insufficient_evidence: Found some evidence but not enough to make a determination
4. Be especially alert for CONTRADICTIONS between different agents' findings.
5. A low verification rate is a valid finding - it means the research needs more depth.
6. Never rubber-stamp claims - genuine verification adds value.

Output your verification results as structured data following the schema exactly."""


def fact_checker(state: DueDiligenceState) -> dict:
    """Fact Checker agent node for LangGraph.

    Runs AFTER all specialist agents to verify their findings.
    """
    start_time = time.time()
    company = state.get("company_name", "Unknown")

    logger.info("[%s] Starting fact-checking for: %s", AGENT_NAME, company)

    try:
        # Step 1: Collect all high-priority claims from specialist agents
        claims = _extract_high_priority_claims(state)

        if not claims:
            return _no_claims_output(company, start_time)

        # Step 2: Independent verification searches
        search = WebSearchTool()
        verification_context = []

        for claim in claims[:10]:  # Cap at 10 claims to stay within budget
            query = f"{company} {claim['title']}"
            results = search.search(query, max_results=3)
            context_str = "\n".join(
                f"  - [{r.title}]({r.url}): {r.snippet[:200]}"
                for r in results
            )
            verification_context.append(
                f"CLAIM by {claim['agent']}: {claim['title']}\n"
                f"Detail: {claim['detail'][:300]}\n"
                f"Verification search results:\n{context_str or '  No results found'}\n"
            )

        # Step 3: LLM verification with structured output
        structured_llm = get_structured_llm(FactCheckReport, temperature=0.0)

        prompt = f"""{SYSTEM_PROMPT}

TARGET COMPANY: {company}

CLAIMS TO VERIFY:
{"".join(f"\\n--- Claim {i+1} ---\\n{vc}" for i, vc in enumerate(verification_context))}

Verify each claim against the search results. Flag any contradictions between agents."""

        report: FactCheckReport = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Step 4: Detect cross-agent contradictions
        contradictions = _detect_contradictions(state, report)

        # Step 5: Build output
        fact_check_results = [{
            "total_checked": report.total_claims_checked,
            "verified": report.verified_count,
            "contradicted": report.contradicted_count,
            "unverifiable": report.unverifiable_count,
            "overall_reliability": report.overall_reliability,
        }]

        flagged = [
            {
                "claim": v.claim,
                "source_agent": v.source_agent,
                "status": v.verification_status,
                "confidence": v.confidence,
                "evidence": v.evidence,
            }
            for v in report.verifications
            if v.verification_status in ("contradicted", "insufficient_evidence")
        ]

        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="fact_verification",
            summary=f"Checked {report.total_claims_checked} claims: "
                    f"{report.verified_count} confirmed, "
                    f"{report.contradicted_count} contradicted, "
                    f"{report.unverifiable_count} unverifiable. "
                    f"Reliability: {report.overall_reliability}",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info(
            "[%s] Completed: %d/%d verified, %d contradictions (%.1fs)",
            AGENT_NAME, report.verified_count, report.total_claims_checked,
            len(contradictions), duration,
        )

        return {
            "fact_check_results": fact_check_results,
            "flagged_claims": flagged,
            "contradictions": contradictions,
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Failed: %s", AGENT_NAME, e)
        return _fallback_output(company, str(e), start_time)


def _extract_high_priority_claims(state: DueDiligenceState) -> list[dict]:
    """Extract the most important claims from all agents for verification."""
    claims = []

    all_findings = (
        state.get("financial_findings", [])
        + state.get("news_findings", [])
        + state.get("competitive_findings", [])
        + state.get("risk_findings", [])
    )

    # Prioritize: high/critical severity, then medium, skip info/low
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

    sorted_findings = sorted(
        all_findings,
        key=lambda f: priority_order.get(f.get("severity", "info"), 4),
    )

    for finding in sorted_findings:
        if finding.get("severity") in ("critical", "high", "medium"):
            claims.append({
                "agent": finding.get("agent", "unknown"),
                "title": finding.get("title", ""),
                "detail": finding.get("detail", ""),
                "severity": finding.get("severity", "medium"),
                "sources": finding.get("sources", []),
            })

    return claims


def _detect_contradictions(state: DueDiligenceState, report: FactCheckReport) -> list[dict]:
    """Detect contradictions between agents' findings."""
    contradictions = []

    # From fact checker's explicit contradiction findings
    for v in report.verifications:
        if v.contradicts_agent and v.verification_status == "contradicted":
            contradictions.append({
                "claim": v.claim,
                "agent_a": v.source_agent,
                "agent_b": v.contradicts_agent,
                "evidence": v.evidence,
                "status": "unresolved",
            })

    # From cross-agent contradiction list
    for c in report.cross_agent_contradictions:
        contradictions.append({
            "claim": c,
            "agent_a": "multiple",
            "agent_b": "multiple",
            "evidence": c,
            "status": "unresolved",
        })

    return contradictions


def _no_claims_output(company: str, start_time: float) -> dict:
    duration = time.time() - start_time
    return {
        "fact_check_results": [{"total_checked": 0, "verified": 0, "contradicted": 0, "unverifiable": 0, "overall_reliability": "no_claims_to_check"}],
        "flagged_claims": [],
        "contradictions": [],
        "pipeline_trace": [
            AgentTrace(
                agent=AGENT_NAME,
                action="fact_verification",
                summary="No high-priority claims to verify",
                duration_seconds=round(duration, 2),
                tokens_used=0,
                cost_usd=0.0,
                error=None,
                retry_count=0,
            )
        ],
    }


def _fallback_output(company: str, error: str, start_time: float) -> dict:
    duration = time.time() - start_time
    return {
        "fact_check_results": [{"error": error, "overall_reliability": "unknown"}],
        "flagged_claims": [],
        "contradictions": [],
        "pipeline_trace": [
            AgentTrace(
                agent=AGENT_NAME,
                action="fact_verification",
                summary=f"FALLBACK: {error}",
                duration_seconds=round(duration, 2),
                tokens_used=0,
                cost_usd=0.0,
                error=error,
                retry_count=0,
            )
        ],
        "errors": [f"[{AGENT_NAME}] {error}"],
        "warnings": ["Fact-checking failed - all findings are UNVERIFIED"],
    }

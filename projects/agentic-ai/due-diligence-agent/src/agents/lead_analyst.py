"""
Lead Analyst Agent - Orchestrator that plans, resolves conflicts, and synthesizes.

Called at THREE points in the pipeline:
1. PLANNING: Decomposes the query into sub-tasks for specialist agents
2. DEBATE: Resolves contradictions flagged by the Fact Checker
3. SYNTHESIS: Produces the final executive summary and report

This is the "brain" of the system - it coordinates all other agents.
"""

from __future__ import annotations

import logging
import time

from src.config import get_agent_config, get_pipeline_config
from src.llm import get_structured_llm, invoke_with_tracking
from src.models.schemas import ResearchPlan, ConflictResolution, ExecutiveSummary
from src.models.state import AgentTrace, DebateEntry, DueDiligenceState

logger = logging.getLogger(__name__)

AGENT_NAME = "lead_analyst"


# ---------------------------------------------------------------------------
# Phase 1: Research Planning
# ---------------------------------------------------------------------------

PLANNING_PROMPT = """You are the Lead Due Diligence Analyst. Your job is to create a research plan
for your team of specialist agents.

Given a company name and research query, decompose the work into specific tasks for each agent:
- financial_analyst: Financial health, revenue, margins, cash flow
- news_sentiment: Recent news, public perception, media coverage
- competitive_intel: Market position, competitors, industry trends
- risk_assessor: Legal, regulatory, operational, reputational risks

Create specific, actionable sub-tasks. Avoid generic instructions.
If the query mentions specific concerns, prioritize those.

TARGET COMPANY: {company}
QUERY: {query}
ANALYSIS DEPTH: {depth}

Create a focused research plan."""


def plan_research(state: DueDiligenceState) -> dict:
    """Phase 1: Create research plan and decompose into agent sub-tasks."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    query = state.get("query", "")
    depth = state.get("analysis_depth", "standard")

    logger.info("[%s] Planning research for: %s", AGENT_NAME, company)

    try:
        structured_llm = get_structured_llm(ResearchPlan, temperature=0.2)

        prompt = PLANNING_PROMPT.format(company=company, query=query, depth=depth)
        plan: ResearchPlan = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Convert plan to agent-keyed dict
        research_plan = {"company_summary": plan.company_summary, "risk_hypothesis": plan.risk_hypothesis}
        for task in plan.sub_tasks:
            research_plan[task.agent] = {
                "focus": task.focus,
                "priority": task.priority,
                "key_questions": task.key_questions,
            }

        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="research_planning",
            summary=f"Created plan with {len(plan.sub_tasks)} sub-tasks, "
                    f"hypothesis: {plan.risk_hypothesis[:100]}",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Plan created: %d tasks (%.1fs)", AGENT_NAME, len(plan.sub_tasks), duration)

        return {
            "research_plan": research_plan,
            "focus_areas": plan.focus_areas,
            "status": "researching",
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Planning failed: %s", AGENT_NAME, e)
        # Fallback: generic plan so pipeline can continue
        return {
            "research_plan": {
                "company_summary": f"Analysis of {company}",
                "risk_hypothesis": "Standard due diligence review",
                "financial_analyst": {"focus": "financial health and performance", "priority": "high", "key_questions": []},
                "news_sentiment": {"focus": "recent news and public perception", "priority": "high", "key_questions": []},
                "competitive_intel": {"focus": "competitive landscape", "priority": "medium", "key_questions": []},
                "risk_assessor": {"focus": "legal, regulatory, and operational risks", "priority": "high", "key_questions": []},
            },
            "focus_areas": ["financial health", "recent news", "competitive position", "risk profile"],
            "status": "researching",
            "pipeline_trace": [
                AgentTrace(
                    agent=AGENT_NAME, action="research_planning",
                    summary=f"FALLBACK plan: {e}", duration_seconds=round(time.time() - start_time, 2),
                    tokens_used=0, cost_usd=0.0, error=str(e), retry_count=0,
                )
            ],
            "warnings": [f"Using fallback research plan: {e}"],
        }


# ---------------------------------------------------------------------------
# Phase 2: Contradiction Debate
# ---------------------------------------------------------------------------

DEBATE_PROMPT = """You are the Lead Due Diligence Analyst resolving contradictions.

Two agents produced conflicting findings. Review both sides and make a judgment call.

COMPANY: {company}

CONTRADICTION:
{contradiction}

Based on the evidence, resolve this contradiction. Consider:
- Which agent's sources are more authoritative?
- Is the contradiction due to different time periods or scopes?
- Could both be partially correct?
- Is there a way to reconcile the findings?

Provide your resolution."""


def resolve_contradictions(state: DueDiligenceState) -> dict:
    """Phase 2: Resolve contradictions between agents via structured debate."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    contradictions = state.get("contradictions", [])
    pipeline_config = get_pipeline_config()
    max_rounds = pipeline_config.get("max_debate_rounds", 2)

    if not contradictions:
        logger.info("[%s] No contradictions to resolve", AGENT_NAME)
        return {
            "status": "reporting",
            "pipeline_trace": [
                AgentTrace(
                    agent=AGENT_NAME, action="debate",
                    summary="No contradictions found - skipping debate",
                    duration_seconds=0.0, tokens_used=0, cost_usd=0.0, error=None, retry_count=0,
                )
            ],
        }

    logger.info("[%s] Resolving %d contradictions", AGENT_NAME, len(contradictions))

    debate_log = []
    resolved_contradictions = []

    try:
        structured_llm = get_structured_llm(ConflictResolution, temperature=0.2)

        for i, contradiction in enumerate(contradictions[:max_rounds]):
            prompt = DEBATE_PROMPT.format(
                company=company,
                contradiction=(
                    f"Agent A ({contradiction.get('agent_a', 'unknown')}): "
                    f"{contradiction.get('claim', '')}\n"
                    f"Agent B ({contradiction.get('agent_b', 'unknown')}): "
                    f"{contradiction.get('evidence', '')}"
                ),
            )

            resolution: ConflictResolution = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

            debate_entry = DebateEntry(
                round_number=i + 1,
                contradiction=resolution.contradiction_summary,
                agent_a=contradiction.get("agent_a", "unknown"),
                agent_b=contradiction.get("agent_b", "unknown"),
                resolution=resolution.resolution,
                confidence=resolution.confidence,
            )
            debate_log.append(debate_entry)

            resolved_contradictions.append({
                **contradiction,
                "status": "resolved",
                "resolution": resolution.resolution,
                "action": resolution.action,
                "confidence": resolution.confidence,
            })

        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="debate",
            summary=f"Resolved {len(debate_log)} contradictions in {len(debate_log)} rounds",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Debate complete: %d resolved (%.1fs)", AGENT_NAME, len(debate_log), duration)

        return {
            "debate_log": debate_log,
            "contradictions": resolved_contradictions,
            "status": "reporting",
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Debate failed: %s", AGENT_NAME, e)
        return {
            "status": "reporting",
            "pipeline_trace": [
                AgentTrace(
                    agent=AGENT_NAME, action="debate",
                    summary=f"FALLBACK: debate skipped due to {e}",
                    duration_seconds=round(time.time() - start_time, 2),
                    tokens_used=0, cost_usd=0.0, error=str(e), retry_count=0,
                )
            ],
            "warnings": [f"Contradiction resolution skipped: {e}. Findings may contain conflicts."],
        }


# ---------------------------------------------------------------------------
# Phase 3: Final Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are the Lead Due Diligence Analyst writing the final report.

Synthesize all agent findings into an executive summary and overall assessment.

COMPANY: {company}
ORIGINAL QUERY: {query}

FINANCIAL FINDINGS ({fin_count}):
{financial_summary}

NEWS & SENTIMENT FINDINGS ({news_count}):
{news_summary}

COMPETITIVE FINDINGS ({comp_count}):
{competitive_summary}

RISK FINDINGS ({risk_count}):
{risk_summary}

FACT-CHECK RESULTS:
{fact_check_summary}

DEBATE RESOLUTIONS:
{debate_summary}

DATA QUALITY NOTE: {data_quality}

Produce a balanced executive summary. Highlight what's known, what's uncertain, and what needs
further investigation. Do not overstate confidence in areas with limited data."""


def synthesize_report(state: DueDiligenceState) -> dict:
    """Phase 3: Produce final executive summary and report."""
    start_time = time.time()
    company = state.get("company_name", "Unknown")
    query = state.get("query", "")

    logger.info("[%s] Synthesizing final report for: %s", AGENT_NAME, company)

    try:
        # Summarize findings from each agent
        financial_summary = _summarize_findings(state.get("financial_findings", []))
        news_summary = _summarize_findings(state.get("news_findings", []))
        competitive_summary = _summarize_findings(state.get("competitive_findings", []))
        risk_summary = _summarize_findings(state.get("risk_findings", []))

        # Fact check summary
        fc_results = state.get("fact_check_results", [{}])
        fc = fc_results[0] if fc_results else {}
        fact_check_summary = (
            f"Checked: {fc.get('total_checked', 0)}, "
            f"Verified: {fc.get('verified', 0)}, "
            f"Contradicted: {fc.get('contradicted', 0)}, "
            f"Reliability: {fc.get('overall_reliability', 'unknown')}"
        )

        # Debate summary
        debate_log = state.get("debate_log", [])
        debate_summary = "\n".join(
            f"- {d.get('contradiction', 'N/A')}: {d.get('resolution', 'N/A')} (confidence: {d.get('confidence', 0)})"
            for d in debate_log
        ) or "No contradictions required resolution."

        # Data quality assessment
        errors = state.get("errors", [])
        data_quality = "High" if not errors else f"Degraded - {len(errors)} agent errors occurred"

        prompt = SYNTHESIS_PROMPT.format(
            company=company,
            query=query,
            financial_summary=financial_summary,
            fin_count=len(state.get("financial_findings", [])),
            news_summary=news_summary,
            news_count=len(state.get("news_findings", [])),
            competitive_summary=competitive_summary,
            comp_count=len(state.get("competitive_findings", [])),
            risk_summary=risk_summary,
            risk_count=len(state.get("risk_findings", [])),
            fact_check_summary=fact_check_summary,
            debate_summary=debate_summary,
            data_quality=data_quality,
        )

        structured_llm = get_structured_llm(ExecutiveSummary, temperature=0.2)
        summary: ExecutiveSummary = invoke_with_tracking(structured_llm, prompt, AGENT_NAME)

        # Build markdown report
        final_report = _build_markdown_report(state, summary)

        duration = time.time() - start_time

        trace = AgentTrace(
            agent=AGENT_NAME,
            action="synthesis",
            summary=f"Report complete: {summary.overall_risk_rating} risk, "
                    f"confidence: {summary.overall_confidence}, "
                    f"recommendation: {summary.recommendation}",
            duration_seconds=round(duration, 2),
            tokens_used=0,
            cost_usd=0.0,
            error=None,
            retry_count=0,
        )

        logger.info("[%s] Synthesis complete: %s (%.1fs)", AGENT_NAME, summary.recommendation, duration)

        return {
            "executive_summary": summary.one_line_verdict,
            "final_report": final_report,
            "overall_risk_rating": summary.overall_risk_rating,
            "overall_confidence": summary.overall_confidence,
            "status": "complete",
            "pipeline_trace": [trace],
        }

    except Exception as e:
        logger.error("[%s] Synthesis failed: %s", AGENT_NAME, e)
        # Build a minimal report from raw findings
        fallback_report = _build_fallback_report(state)
        return {
            "executive_summary": f"Analysis of {company} completed with errors. See findings below.",
            "final_report": fallback_report,
            "overall_risk_rating": "unknown",
            "overall_confidence": 0.0,
            "status": "complete",
            "pipeline_trace": [
                AgentTrace(
                    agent=AGENT_NAME, action="synthesis",
                    summary=f"FALLBACK report: {e}",
                    duration_seconds=round(time.time() - start_time, 2),
                    tokens_used=0, cost_usd=0.0, error=str(e), retry_count=0,
                )
            ],
            "errors": [f"[{AGENT_NAME}] Synthesis failed: {e}"],
        }


def _summarize_findings(findings: list) -> str:
    """Create a compact text summary of findings for the synthesis prompt."""
    if not findings:
        return "No findings available."

    lines = []
    for f in findings[:15]:  # Cap to avoid prompt bloat
        sev = f.get("severity", "info")
        title = f.get("title", "Untitled")
        detail = f.get("detail", "")[:200]
        lines.append(f"[{sev.upper()}] {title}\n  {detail}")

    return "\n\n".join(lines)


def _build_markdown_report(state: DueDiligenceState, summary: ExecutiveSummary) -> str:
    """Build a comprehensive markdown report."""
    company = state.get("company_name", "Unknown")

    sections = [
        f"# Due Diligence Report: {company}",
        f"",
        f"## Executive Summary",
        f"",
        f"**Verdict:** {summary.one_line_verdict}",
        f"",
        f"**Risk Rating:** {summary.overall_risk_rating.upper()} | "
        f"**Confidence:** {summary.overall_confidence:.0%} | "
        f"**Recommendation:** {summary.recommendation}",
        f"",
        f"### Key Strengths",
        *[f"- {s}" for s in summary.key_strengths],
        f"",
        f"### Key Risks",
        *[f"- {r}" for r in summary.key_risks],
        f"",
    ]

    if summary.key_uncertainties:
        sections.extend([
            f"### Key Uncertainties",
            *[f"- {u}" for u in summary.key_uncertainties],
            f"",
        ])

    if summary.action_items:
        sections.extend([
            f"### Recommended Next Steps",
            *[f"{i+1}. {a}" for i, a in enumerate(summary.action_items)],
            f"",
        ])

    # Detailed findings sections
    for category, key, title in [
        ("financial", "financial_findings", "Financial Analysis"),
        ("news", "news_findings", "News & Sentiment"),
        ("competitive", "competitive_findings", "Competitive Intelligence"),
        ("risk", "risk_findings", "Risk Assessment"),
    ]:
        findings = state.get(key, [])
        if findings:
            sections.extend([f"---", f"", f"## {title}", f""])
            for f in findings:
                sev = f.get("severity", "info")
                icon = {"critical": "!!!", "high": "!!", "medium": "!", "low": "-", "info": "-"}.get(sev, "-")
                sections.append(f"### {icon} {f.get('title', 'Untitled')}")
                sections.append(f"")
                sections.append(f"{f.get('detail', '')}")
                if f.get("sources"):
                    sections.append(f"")
                    sections.append(f"Sources: {', '.join(f.get('sources', []))}")
                conf = f.get("confidence", 0)
                verified = "Verified" if f.get("verified") else "Unverified"
                sections.append(f"")
                sections.append(f"*Confidence: {conf:.0%} | {verified} | Severity: {sev}*")
                sections.append(f"")

    # Fact-check section
    fc_results = state.get("fact_check_results", [])
    if fc_results:
        fc = fc_results[0] if fc_results else {}
        sections.extend([
            f"---", f"",
            f"## Fact-Check Summary", f"",
            f"- Claims checked: {fc.get('total_checked', 0)}",
            f"- Verified: {fc.get('verified', 0)}",
            f"- Contradicted: {fc.get('contradicted', 0)}",
            f"- Unverifiable: {fc.get('unverifiable', 0)}",
            f"- Overall reliability: {fc.get('overall_reliability', 'unknown')}",
            f"",
        ])

    # Pipeline trace
    traces = state.get("pipeline_trace", [])
    if traces:
        total_duration = sum(t.get("duration_seconds", 0) for t in traces)
        sections.extend([
            f"---", f"",
            f"## Pipeline Execution Log", f"",
            f"| Agent | Action | Duration | Status |",
            f"|-------|--------|----------|--------|",
        ])
        for t in traces:
            err = "Error" if t.get("error") else "OK"
            sections.append(
                f"| {t.get('agent', '?')} | {t.get('action', '?')} | "
                f"{t.get('duration_seconds', 0):.1f}s | {err} |"
            )
        sections.extend([f"", f"**Total execution time:** {total_duration:.1f}s", f""])

    # Errors and warnings
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    if errors or warnings:
        sections.extend([f"---", f"", f"## Caveats", f""])
        for e in errors:
            sections.append(f"- ERROR: {e}")
        for w in warnings:
            sections.append(f"- WARNING: {w}")
        sections.append(f"")

    sections.extend([
        f"---",
        f"*Report generated by Due Diligence Agent. "
        f"This is an automated analysis and should not be used as the sole basis for business decisions. "
        f"Independent verification of key findings is recommended.*",
    ])

    return "\n".join(sections)


def _build_fallback_report(state: DueDiligenceState) -> str:
    """Build a minimal report when synthesis LLM fails."""
    company = state.get("company_name", "Unknown")
    lines = [
        f"# Due Diligence Report: {company}",
        f"",
        f"**NOTE: Automated synthesis failed. Raw findings listed below.**",
        f"",
    ]

    for key, title in [
        ("financial_findings", "Financial"),
        ("news_findings", "News & Sentiment"),
        ("competitive_findings", "Competitive"),
        ("risk_findings", "Risk"),
    ]:
        findings = state.get(key, [])
        if findings:
            lines.append(f"## {title}")
            for f in findings:
                lines.append(f"- [{f.get('severity', '?')}] {f.get('title', 'Untitled')}: {f.get('detail', '')[:200]}")
            lines.append("")

    return "\n".join(lines)

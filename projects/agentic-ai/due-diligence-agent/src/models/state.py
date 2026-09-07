"""
Pipeline state schema - the shared data contract between all agents.

Every agent reads from and writes to this TypedDict. LangGraph manages
state merging via reducer annotations (operator.add for list fields).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentFinding(TypedDict, total=False):
    """A single finding from any agent."""
    agent: str                      # Which agent produced this
    category: str                   # financial | news | competitive | risk
    title: str                      # Short headline
    detail: str                     # Full explanation
    severity: str                   # critical | high | medium | low | info
    confidence: float               # 0.0 to 1.0
    sources: list[str]              # URLs or document references
    verified: bool                  # True if fact-checker confirmed
    verification_notes: str         # Fact-checker's notes


class AgentTrace(TypedDict, total=False):
    """Execution trace for a single agent step."""
    agent: str
    action: str                     # e.g., "search", "analyze", "verify"
    summary: str
    duration_seconds: float
    tokens_used: int
    cost_usd: float
    error: str | None
    retry_count: int


class DebateEntry(TypedDict, total=False):
    """A single round in the contradiction resolution debate."""
    round_number: int
    contradiction: str              # The conflicting claims
    agent_a: str                    # First agent involved
    agent_b: str                    # Second agent involved
    resolution: str                 # Lead analyst's resolution
    confidence: float


class DueDiligenceState(TypedDict, total=False):
    """Complete pipeline state - passed through all LangGraph nodes."""

    # --- Input ---
    company_name: str               # Target company/entity
    query: str                      # Original user query (may include specific focus areas)
    analysis_depth: str             # quick | standard | deep

    # --- Planning (Lead Analyst output) ---
    research_plan: dict             # Decomposed sub-tasks per agent
    focus_areas: list[str]          # Specific areas to investigate

    # --- Agent Outputs (append-only via operator.add) ---
    financial_findings: Annotated[list[AgentFinding], operator.add]
    news_findings: Annotated[list[AgentFinding], operator.add]
    competitive_findings: Annotated[list[AgentFinding], operator.add]
    risk_findings: Annotated[list[AgentFinding], operator.add]

    # --- Fact Checking ---
    fact_check_results: Annotated[list[dict], operator.add]
    flagged_claims: Annotated[list[dict], operator.add]

    # --- Debate / Conflict Resolution ---
    contradictions: Annotated[list[dict], operator.add]
    debate_log: Annotated[list[DebateEntry], operator.add]

    # --- Final Report ---
    executive_summary: str
    final_report: str               # Full markdown report
    overall_risk_rating: str        # low | moderate | high | critical
    overall_confidence: float       # Weighted average confidence

    # --- Execution Metadata ---
    pipeline_trace: Annotated[list[AgentTrace], operator.add]
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    total_tokens_used: int
    total_cost_usd: float
    status: str                     # planning | researching | fact_checking | debating | reporting | complete | failed


def default_state(company_name: str, query: str = "", depth: str = "standard") -> DueDiligenceState:
    """Create a fresh pipeline state with safe defaults."""
    return DueDiligenceState(
        company_name=company_name,
        query=query or f"Comprehensive due diligence analysis of {company_name}",
        analysis_depth=depth,
        research_plan={},
        focus_areas=[],
        financial_findings=[],
        news_findings=[],
        competitive_findings=[],
        risk_findings=[],
        fact_check_results=[],
        flagged_claims=[],
        contradictions=[],
        debate_log=[],
        executive_summary="",
        final_report="",
        overall_risk_rating="unknown",
        overall_confidence=0.0,
        pipeline_trace=[],
        errors=[],
        warnings=[],
        total_tokens_used=0,
        total_cost_usd=0.0,
        status="planning",
    )

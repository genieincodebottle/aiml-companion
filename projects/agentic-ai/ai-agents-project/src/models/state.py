# ============================================
# Research State Schema & Pydantic Models
# ============================================
# Defines the shared state flowing through the LangGraph pipeline,
# plus structured output schemas for agents that use with_structured_output().

import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field


# === Shared State (flows through entire graph) ===
class ResearchState(TypedDict):
    """Typed state shared across all agents in the research pipeline.

    Uses Annotated + operator.add for fields that accumulate across
    parallel Send() executions (sources, search_queries_used, errors, pipeline_trace).
    """
    # Input
    query: str

    # Planner output
    sub_topics: list[str]
    research_plan: str

    # Researcher output (parallel via Send(), merged with operator.add)
    sources: Annotated[list[dict], operator.add]
    search_queries_used: Annotated[list[str], operator.add]

    # Quality Gate output
    quality_score: float
    quality_passed: bool

    # Analyst output
    key_claims: list[dict]
    conflicts: list[dict]

    # Synthesizer output
    synthesis: str
    source_ranking: list[dict]

    # Writer output (versioned)
    drafts: list[dict]
    current_draft: str

    # Reviewer output
    review: dict
    revision_count: int

    # Pipeline metadata (token_count uses operator.add for parallel fan-out)
    token_count: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]
    final_report: str
    pipeline_trace: Annotated[list[dict], operator.add]


def default_state(query: str) -> dict:
    """Create a fresh state dict with all fields initialized."""
    return {
        "query": query,
        "sub_topics": [],
        "research_plan": "",
        "sources": [],
        "search_queries_used": [],
        "quality_score": 0.0,
        "quality_passed": False,
        "key_claims": [],
        "conflicts": [],
        "synthesis": "",
        "source_ranking": [],
        "drafts": [],
        "current_draft": "",
        "review": {},
        "revision_count": 0,
        "token_count": 0,
        "errors": [],
        "final_report": "",
        "pipeline_trace": [],
    }


# === Pydantic Schemas for Structured Output ===

class PlannerOutput(BaseModel):
    """Structured output from the Planner agent."""
    sub_topics: list[str] = Field(
        description="1-3 focused sub-topics to research independently",
        min_length=1,
        max_length=3,
    )
    research_plan: str = Field(
        description="Brief strategy description (2-3 sentences)",
    )


class ClaimOutput(BaseModel):
    """A single extracted claim with evidence."""
    claim: str = Field(description="The factual claim")
    source_idx: int = Field(description="1-based index of the supporting source")
    confidence: str = Field(description="high, medium, or low")
    evidence: str = Field(description="Quote or paraphrase from the source supporting this claim")


class AnalystOutput(BaseModel):
    """Structured output from the Analyst agent."""
    claims: list[ClaimOutput] = Field(description="5-8 extracted claims with evidence")
    conflicts: list[str] = Field(
        default_factory=list,
        description="Cross-source contradictions found (if any)",
    )


class ReviewOutput(BaseModel):
    """Structured output from the Reviewer agent."""
    score: int = Field(description="Overall quality score 1-10", ge=1, le=10)
    issues: list[str] = Field(
        default_factory=list,
        description="Specific issues to fix (unsupported claims, weak sections, etc.)",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete improvement suggestions for the writer",
    )
    passed: bool = Field(description="True if score >= 7 and report is publication-ready")

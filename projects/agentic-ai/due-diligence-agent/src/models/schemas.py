"""
Pydantic schemas for structured LLM output.

Every LLM call uses .with_structured_output(Schema) to guarantee
valid JSON responses. This eliminates parsing errors and retries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Lead Analyst Schemas
# ---------------------------------------------------------------------------

class ResearchSubTask(BaseModel):
    """A single research sub-task assigned to a specialist agent."""
    agent: str = Field(description="Agent to assign: financial_analyst | news_sentiment | competitive_intel | risk_assessor")
    focus: str = Field(description="Specific aspect to investigate")
    priority: str = Field(default="medium", description="high | medium | low")
    key_questions: list[str] = Field(default_factory=list, description="2-4 specific questions to answer")


class ResearchPlan(BaseModel):
    """Lead Analyst's research decomposition."""
    company_summary: str = Field(description="Brief known context about the company")
    sub_tasks: list[ResearchSubTask] = Field(description="Research tasks for each specialist")
    focus_areas: list[str] = Field(description="Key focus areas identified from the query")
    risk_hypothesis: str = Field(description="Initial risk hypothesis to test")


class ConflictResolution(BaseModel):
    """Lead Analyst's resolution of contradicting findings."""
    contradiction_summary: str = Field(description="What the agents disagree on")
    resolution: str = Field(description="The resolved position with reasoning")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the resolution")
    action: str = Field(description="keep_both | prefer_agent_a | prefer_agent_b | discard_both | needs_more_research")


# ---------------------------------------------------------------------------
# Financial Analyst Schemas
# ---------------------------------------------------------------------------

class FinancialMetric(BaseModel):
    """A single financial data point."""
    metric_name: str = Field(description="e.g., Revenue, Net Income, Gross Margin")
    value: str = Field(description="The value with units")
    trend: str = Field(description="increasing | decreasing | stable | volatile | unknown")
    assessment: str = Field(description="positive | negative | neutral | concerning")
    source: str = Field(description="URL or document reference")


class FinancialAnalysis(BaseModel):
    """Complete financial analyst output."""
    company_name: str
    financial_health_rating: str = Field(description="strong | moderate | weak | critical | insufficient_data")
    key_metrics: list[FinancialMetric] = Field(default_factory=list)
    revenue_analysis: str = Field(default="", description="Revenue trends and growth analysis")
    profitability_analysis: str = Field(default="", description="Margin and profitability assessment")
    cash_flow_notes: str = Field(default="", description="Cash flow observations if available")
    red_flags: list[str] = Field(default_factory=list, description="Financial warning signs")
    green_flags: list[str] = Field(default_factory=list, description="Financial strengths")
    data_gaps: list[str] = Field(default_factory=list, description="Missing data that limits analysis")
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# News & Sentiment Schemas
# ---------------------------------------------------------------------------

class NewsEvent(BaseModel):
    """A significant news event."""
    date: str = Field(description="Date or date range")
    headline: str
    sentiment: str = Field(description="positive | negative | neutral | mixed")
    impact: str = Field(description="high | medium | low")
    source: str
    summary: str


class NewsSentimentAnalysis(BaseModel):
    """Complete news and sentiment output."""
    company_name: str
    overall_sentiment: str = Field(description="positive | negative | neutral | mixed")
    sentiment_trend: str = Field(description="improving | declining | stable | volatile")
    key_events: list[NewsEvent] = Field(default_factory=list)
    public_perception: str = Field(default="", description="Summary of public/market perception")
    media_coverage_volume: str = Field(default="unknown", description="high | moderate | low | minimal")
    social_media_notes: str = Field(default="", description="Social media sentiment if found")
    potential_concerns: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Competitive Intelligence Schemas
# ---------------------------------------------------------------------------

class Competitor(BaseModel):
    """A single competitor profile."""
    name: str
    market_position: str = Field(description="leader | challenger | follower | niche")
    key_strengths: list[str] = Field(default_factory=list)
    key_weaknesses: list[str] = Field(default_factory=list)
    estimated_market_share: str = Field(default="unknown")


class CompetitiveAnalysis(BaseModel):
    """Complete competitive intelligence output."""
    company_name: str
    industry: str
    market_position: str = Field(description="Company's market position")
    competitors: list[Competitor] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list, description="Company's moats")
    competitive_risks: list[str] = Field(default_factory=list, description="Competitive threats")
    market_trends: list[str] = Field(default_factory=list)
    differentiation_summary: str = Field(default="")
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk Assessor Schemas
# ---------------------------------------------------------------------------

class RiskItem(BaseModel):
    """A single identified risk."""
    risk_type: str = Field(description="legal | regulatory | operational | reputational | financial | strategic | technology")
    title: str
    description: str
    severity: str = Field(description="critical | high | medium | low")
    likelihood: str = Field(description="very_likely | likely | possible | unlikely | rare")
    mitigation: str = Field(default="", description="Known mitigations or recommended actions")
    source: str = Field(default="")


class RiskAssessment(BaseModel):
    """Complete risk assessment output."""
    company_name: str
    overall_risk_level: str = Field(description="critical | high | moderate | low")
    risks: list[RiskItem] = Field(default_factory=list)
    regulatory_environment: str = Field(default="", description="Summary of regulatory landscape")
    legal_history: str = Field(default="", description="Known legal issues or litigation")
    esg_concerns: str = Field(default="", description="Environmental, social, governance notes")
    risk_summary: str = Field(default="", description="2-3 sentence risk overview")
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact Checker Schemas
# ---------------------------------------------------------------------------

class ClaimVerification(BaseModel):
    """Verification result for a single claim."""
    claim: str = Field(description="The original claim being verified")
    source_agent: str = Field(description="Which agent made this claim")
    verification_status: str = Field(description="confirmed | partially_confirmed | unverifiable | contradicted | insufficient_evidence")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Supporting or contradicting evidence found")
    contradicts_agent: str = Field(default="", description="If contradicted, which other agent's finding conflicts")
    source: str = Field(default="")


class FactCheckReport(BaseModel):
    """Complete fact-checking output."""
    total_claims_checked: int
    verified_count: int
    contradicted_count: int
    unverifiable_count: int
    verifications: list[ClaimVerification] = Field(default_factory=list)
    cross_agent_contradictions: list[str] = Field(default_factory=list, description="Claims where agents disagree")
    overall_reliability: str = Field(default="moderate", description="high | moderate | low")
    notes: str = Field(default="")


# ---------------------------------------------------------------------------
# Executive Summary Schema
# ---------------------------------------------------------------------------

class ExecutiveSummary(BaseModel):
    """Final executive summary produced by Lead Analyst."""
    company_name: str
    one_line_verdict: str = Field(description="Single sentence overall assessment")
    overall_risk_rating: str = Field(description="low | moderate | high | critical")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    key_strengths: list[str] = Field(description="Top 3-5 strengths")
    key_risks: list[str] = Field(description="Top 3-5 risks")
    key_uncertainties: list[str] = Field(default_factory=list, description="Areas where data was insufficient")
    recommendation: str = Field(description="proceed | proceed_with_caution | further_investigation_needed | high_risk_avoid")
    action_items: list[str] = Field(default_factory=list, description="Recommended next steps")

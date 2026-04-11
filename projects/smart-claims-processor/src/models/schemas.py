"""
Pydantic v2 schemas for structured LLM outputs.
Every agent returns one of these - zero parsing errors guaranteed.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ClaimType(str, Enum):
    AUTO_COLLISION = "auto_collision"
    AUTO_THEFT = "auto_theft"
    PROPERTY_FIRE = "property_fire"
    PROPERTY_WATER = "property_water"
    LIABILITY = "liability"
    MEDICAL = "medical"
    UNKNOWN = "unknown"


class ClaimDecision(str, Enum):
    APPROVED = "approved"
    APPROVED_PARTIAL = "approved_partial"
    DENIED = "denied"
    PENDING_DOCUMENTS = "pending_documents"
    ESCALATED_HITL = "escalated_human_review"
    FRAUD_INVESTIGATION = "fraud_investigation"
    AUTO_REJECTED = "auto_rejected"


class FraudRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CONFIRMED = "confirmed"


class HITLPriority(str, Enum):
    CRITICAL = "critical"   # Resolve within 4 hours
    HIGH = "high"           # Resolve within 24 hours
    NORMAL = "normal"       # Resolve within 72 hours


class CoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    NEEDS_VERIFICATION = "needs_verification"


# ─────────────────────────────────────────────
# Agent Output Schemas
# ─────────────────────────────────────────────

class IntakeValidationOutput(BaseModel):
    """Output from Claims Intake Agent."""
    is_valid: bool = Field(description="Whether the claim passes initial validation")
    claim_type: ClaimType = Field(description="Detected claim type")
    policy_active: bool = Field(description="Whether the policy is active and not lapsed")
    claimant_eligible: bool = Field(description="Whether the claimant is eligible to file")
    missing_documents: list[str] = Field(default_factory=list, description="List of required but missing documents")
    intake_notes: str = Field(description="Summary of intake findings")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    validation_flags: list[str] = Field(default_factory=list, description="Any flags raised during intake")


class FraudPatternOutput(BaseModel):
    """Output from CrewAI Pattern Analyst."""
    pattern_matches: list[str] = Field(description="Fraud patterns matched from database")
    risk_indicators: list[str] = Field(description="Specific fraud indicators detected")
    pattern_score: float = Field(ge=0.0, le=1.0, description="Pattern-based fraud probability 0-1")
    analysis: str = Field(description="Detailed pattern analysis")


class AnomalyDetectionOutput(BaseModel):
    """Output from CrewAI Anomaly Detector."""
    statistical_anomalies: list[str] = Field(description="Statistical outliers detected")
    claim_frequency_flag: bool = Field(description="True if claimant has unusually high claim frequency")
    amount_anomaly: bool = Field(description="True if amount deviates significantly from similar claims")
    timing_anomaly: bool = Field(description="True if claim timing is suspicious (e.g. policy just purchased)")
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Anomaly-based fraud probability 0-1")
    analysis: str = Field(description="Detailed anomaly analysis")


class SocialValidationOutput(BaseModel):
    """Output from CrewAI Social Validator."""
    story_consistent: bool = Field(description="Whether the claimant's account is internally consistent")
    inconsistencies: list[str] = Field(description="Specific inconsistencies found in the claim narrative")
    identity_flags: list[str] = Field(description="Identity verification concerns")
    validation_score: float = Field(ge=0.0, le=1.0, description="Story consistency score 0-1 (higher = more consistent)")
    analysis: str = Field(description="Detailed validation analysis")


class FraudAssessmentOutput(BaseModel):
    """Final output from the full CrewAI Fraud Detection Crew."""
    fraud_risk_level: FraudRiskLevel = Field(description="Overall fraud risk classification")
    fraud_score: float = Field(ge=0.0, le=1.0, description="Composite fraud probability 0-1")
    primary_concerns: list[str] = Field(description="Top fraud concerns to highlight")
    recommendation: str = Field(description="Recommended action: proceed | escalate | reject")
    crew_summary: str = Field(description="Synthesized findings from all crew members")
    pattern_score: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)


class DamageAssessmentOutput(BaseModel):
    """Output from Damage Assessment Agent."""
    assessed_damage_usd: float = Field(ge=0.0, description="Total assessed damage in USD")
    line_items: list[dict] = Field(description="Itemized damage breakdown")
    repair_vs_replace: str = Field(description="Recommendation: repair | replace | total_loss")
    assessment_confidence: float = Field(ge=0.0, le=1.0)
    assessment_notes: str = Field(description="Assessor's notes and methodology")
    requires_physical_inspection: bool = Field(description="Whether a physical inspector should visit")
    comparable_claims_avg: Optional[float] = Field(None, description="Average payout for comparable claims")


class PolicyCheckOutput(BaseModel):
    """Output from Policy Compliance Agent."""
    coverage_status: CoverageStatus
    covered_amount_usd: float = Field(ge=0.0, description="Amount covered by policy")
    deductible_usd: float = Field(ge=0.0)
    exclusions_triggered: list[str] = Field(default_factory=list, description="Policy exclusions that apply")
    coverage_notes: str = Field(description="Explanation of coverage determination")
    compliance_flags: list[str] = Field(default_factory=list, description="Regulatory compliance concerns")
    policy_limits: dict = Field(description="Applicable policy limits")
    confidence: float = Field(ge=0.0, le=1.0)


class SettlementOutput(BaseModel):
    """Output from Settlement Calculator Agent."""
    decision: ClaimDecision
    settlement_amount_usd: float = Field(ge=0.0)
    gross_damage_usd: float = Field(ge=0.0, description="Damage before deductions")
    deductible_applied_usd: float = Field(ge=0.0)
    depreciation_applied_usd: float = Field(ge=0.0)
    calculation_breakdown: list[str] = Field(description="Step-by-step calculation, max 6 steps")
    denial_reasons: list[str] = Field(default_factory=list, description="Reasons if denied")
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    regulatory_compliance: bool = Field(default=True, description="Passes state regulatory requirements")


class EvaluationOutput(BaseModel):
    """Output from LLM-as-Judge Evaluator."""
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall decision quality 0-1")
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    fairness_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    transparency_score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(description="Whether the decision passes quality gate")
    feedback: str = Field(description="Specific improvement recommendations if failed")
    flags: list[str] = Field(default_factory=list, description="Critical issues requiring immediate attention")


class CommunicationOutput(BaseModel):
    """Output from Communication Agent."""
    subject: str = Field(description="Email/notification subject line")
    message: str = Field(description="Full notification message to claimant")
    internal_notes: str = Field(description="Internal adjuster notes (not sent to claimant)")
    next_steps: list[str] = Field(description="Action items for claimant")
    appeal_instructions: Optional[str] = Field(None, description="How to appeal if denied")

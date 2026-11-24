"""Pydantic schemas shared across the platform planes.

Two families live here:

  1. Platform primitives - the gateway request/response, policy decisions, agent
     cards, prompt versions. These are how the control plane talks to itself.
  2. Tenant structured outputs - the validated objects the fincrime squad's agents
     must return. Every agent is bound to a schema so the runtime never parses free
     text; a malformed model response is a caught error, not a silent corruption.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Gateway primitives
# --------------------------------------------------------------------------- #
class ModelTier(str, Enum):
    """Routing tiers. The gateway maps a tier -> a concrete model per policy."""

    CHEAP = "cheap"
    PREMIUM = "premium"


class GatewayUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "offline"
    cached: bool = False
    provider: Literal["offline", "gemini"] = "offline"


# --------------------------------------------------------------------------- #
# Policy engine
# --------------------------------------------------------------------------- #
class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyDecision(BaseModel):
    """The result of evaluating a tool call against the policy engine."""

    effect: PolicyEffect
    rule: str  # which rule fired, for the audit trail
    reason: str


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #
class PromptVersion(BaseModel):
    """One immutable, versioned entry in the prompt registry."""

    name: str
    version: int
    template: str
    labels: list[str] = Field(default_factory=list)  # e.g. ["production", "candidate"]


class AgentCard(BaseModel):
    """An A2A-style capability card. How an agent declares itself to the platform.

    The card is the contract: which tools the agent may request, which model tier it
    runs on, and whether its actions are side-effecting. The policy engine reads it;
    the runtime enforces it.
    """

    name: str
    role: str
    description: str
    model_tier: ModelTier = ModelTier.PREMIUM
    allowed_tools: list[str] = Field(default_factory=list)
    side_effecting: bool = False


# --------------------------------------------------------------------------- #
# Tenant structured outputs (fincrime investigation squad)
# --------------------------------------------------------------------------- #
class TriageResult(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    is_actionable: bool
    rationale: str


class GraphFinding(BaseModel):
    related_entities: list[str] = Field(default_factory=list)
    suspicious_patterns: list[str] = Field(default_factory=list)
    risk_note: str


class ScreeningResult(BaseModel):
    sanctions_hit: bool
    pep_hit: bool
    matched_lists: list[str] = Field(default_factory=list)
    note: str


class EvidenceItem(BaseModel):
    """Every narrative claim must cite one of these. No citation -> fail closed."""

    source: str  # tool that produced it, e.g. "graph.expand" or "screening.check"
    detail: str


class NarrativeDraft(BaseModel):
    summary: str
    claims: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_action: Literal["file_sar", "close", "escalate"]


class CritiqueResult(BaseModel):
    approved: bool
    every_claim_cited: bool
    issues: list[str] = Field(default_factory=list)
    note: str


class JudgeVerdict(BaseModel):
    """An LLM-as-judge score over a completed investigation (eval plane)."""

    score: float = Field(ge=0.0, le=1.0)
    grounded: bool  # is every narrative claim cited?
    disposition_sound: bool  # is the recommended action defensible from the evidence?
    rationale: str


# --------------------------------------------------------------------------- #
# Case + platform run views (API surface)
# --------------------------------------------------------------------------- #
class Alert(BaseModel):
    """The synthetic AML alert that seeds an investigation workflow."""

    id: str
    entity_id: str
    entity_name: str
    alert_type: str
    amount_usd: float
    description: str


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    COMPENSATED = "compensated"


class WorkflowView(BaseModel):
    """A snapshot of a workflow the API/console renders."""

    workflow_id: str
    tenant: str
    alert_id: str
    status: WorkflowStatus
    step_count: int = 0
    cost_usd: float = 0.0
    disposition: str | None = None
    pending_action: dict[str, Any] | None = None
    # The audit trail: who opened the case, who decided it, and why.
    started_by: str | None = None
    decided_by: str | None = None
    decision: str | None = None
    decision_note: str | None = None
    decided_at: float | None = None
    dead_letter_reason: str | None = None

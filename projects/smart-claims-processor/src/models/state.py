"""
LangGraph state definition for the Smart Claims Processor.

Uses TypedDict with Annotated lists for append-only parallel-safe state.
All agent outputs are typed; missing fields default to None.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from src.models.schemas import (
    ClaimDecision,
    DamageAssessmentOutput,
    EvaluationOutput,
    FraudAssessmentOutput,
    HITLPriority,
    IntakeValidationOutput,
    PolicyCheckOutput,
    SettlementOutput,
    CommunicationOutput,
)


class ClaimInput(TypedDict):
    """Raw claim data as submitted by claimant."""
    claim_id: str
    policy_number: str
    claimant_name: str
    claimant_email: str
    claimant_phone: str
    claimant_dob: str
    incident_date: str
    incident_type: str
    incident_description: str
    incident_location: str
    police_report_number: Optional[str]
    estimated_amount: float
    vehicle_year: Optional[int]
    vehicle_make: Optional[str]
    vehicle_model: Optional[str]
    documents: list[str]
    is_appeal: bool
    original_claim_id: Optional[str]    # Set when this is an appeal


class ClaimsState(TypedDict):
    """
    Full pipeline state. Fields are populated incrementally as agents run.
    append_log uses operator.add so parallel agents can each push trace entries.
    """

    # ── Input ────────────────────────────────────────────────────────────
    claim: ClaimInput
    masked_claim: Optional[dict]        # PII-masked version sent to LLMs

    # ── Intake ───────────────────────────────────────────────────────────
    intake_output: Optional[IntakeValidationOutput]

    # ── Fraud Detection (CrewAI) ─────────────────────────────────────────
    fraud_output: Optional[FraudAssessmentOutput]

    # ── Damage Assessment ────────────────────────────────────────────────
    damage_output: Optional[DamageAssessmentOutput]

    # ── Policy Check ─────────────────────────────────────────────────────
    policy_output: Optional[PolicyCheckOutput]

    # ── Settlement ───────────────────────────────────────────────────────
    settlement_output: Optional[SettlementOutput]

    # ── Evaluation ───────────────────────────────────────────────────────
    evaluation_output: Optional[EvaluationOutput]
    evaluation_passed: Optional[bool]

    # ── Communication ────────────────────────────────────────────────────
    communication_output: Optional[CommunicationOutput]

    # ── HITL ─────────────────────────────────────────────────────────────
    hitl_required: bool
    hitl_triggers: list[str]            # Reasons HITL was triggered
    hitl_priority: Optional[HITLPriority]
    hitl_priority_score: Optional[float]
    hitl_ticket_id: Optional[str]
    human_decision: Optional[str]       # Decision entered by human reviewer
    human_reviewer_id: Optional[str]
    human_notes: Optional[str]
    human_override: bool                # True if human overrode AI recommendation

    # ── Final Decision ───────────────────────────────────────────────────
    final_decision: Optional[ClaimDecision]
    final_amount_usd: Optional[float]

    # ── Guardrails ───────────────────────────────────────────────────────
    guardrails_passed: bool
    guardrails_violations: list[str]
    agent_call_count: int
    total_tokens_used: int
    total_cost_usd: float
    execution_start_time: Optional[str]

    # ── Audit & Tracing ──────────────────────────────────────────────────
    pipeline_trace: Annotated[list[dict], operator.add]   # Append-only log
    error_log: Annotated[list[str], operator.add]         # Append-only errors


def initial_state(claim: ClaimInput) -> ClaimsState:
    """Build a fresh state from a raw claim input.

    Most fields start as None (TypedDict Optional fields). Only non-None
    defaults and required fields are listed explicitly.
    """
    state: dict = {
        "claim": claim,
        # Booleans that default to False
        "hitl_required": False,
        "human_override": False,
        "guardrails_passed": True,
        # Numeric defaults
        "agent_call_count": 0,
        "total_tokens_used": 0,
        "total_cost_usd": 0.0,
        # Append-only lists (operator.add needs a starting list)
        "hitl_triggers": [],
        "guardrails_violations": [],
        "pipeline_trace": [],
        "error_log": [],
    }
    # All remaining Optional fields default to None
    for key in ClaimsState.__annotations__:
        if key not in state:
            state[key] = None
    return ClaimsState(**state)

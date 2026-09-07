"""
HITL (Human-In-The-Loop) Checkpoint - determines whether a claim needs human review.

Trigger logic:
  1. Check each trigger rule against current state
  2. Compute priority score (0-100) using weighted factors
  3. Classify priority: critical (>=80), high (60-79), normal (<60)
  4. Return list of triggered reasons + priority

This is a pure function - no LLM calls, no side effects.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import get_hitl_config
from src.models.schemas import FraudAssessmentOutput, HITLPriority, IntakeValidationOutput

logger = logging.getLogger(__name__)


def check_hitl_required(
    claim: dict,
    intake_output: Optional[IntakeValidationOutput],
    fraud_output: Optional[FraudAssessmentOutput],
    damage_assessed_usd: float = 0.0,
    claim_history_count: int = 0,
    agent_confidence_scores: Optional[list[float]] = None,
) -> tuple[bool, list[str], HITLPriority, float]:
    """
    Evaluate all HITL triggers.

    Returns:
        (requires_hitl, trigger_reasons, priority, priority_score)
    """
    cfg = get_hitl_config()
    triggers_cfg = cfg["triggers"]
    weights = cfg["priority_weights"]

    from src.utils import currency_symbol
    _sym = currency_symbol()

    triggers: list[str] = []
    priority_components: dict[str, float] = {}

    estimated_amount = float(claim.get("estimated_amount", 0))
    is_appeal = claim.get("is_appeal", False)

    # ── Trigger 1: High claim amount ──────────────────────────────────────────
    min_amount = triggers_cfg.get("min_amount", 10000)
    if estimated_amount >= min_amount:
        triggers.append(f"Claim amount {_sym}{estimated_amount:,.0f} exceeds review threshold {_sym}{min_amount:,.0f}")
        priority_components["amount"] = min(estimated_amount / min_amount / 5, 1.0)
    else:
        priority_components["amount"] = 0.0

    # ── Trigger 2: Fraud score ────────────────────────────────────────────────
    fraud_threshold = triggers_cfg.get("fraud_score", 0.65)
    fraud_score = 0.0
    if fraud_output:
        fraud_score = fraud_output.fraud_score
        if fraud_score >= fraud_threshold:
            triggers.append(
                f"Fraud score {fraud_score:.2f} exceeds threshold {fraud_threshold:.2f} "
                f"(risk: {fraud_output.fraud_risk_level.value})"
            )
        priority_components["fraud_score"] = fraud_score
    else:
        priority_components["fraud_score"] = 0.0

    # ── Trigger 3: Low agent confidence ──────────────────────────────────────
    low_conf_threshold = triggers_cfg.get("low_confidence", 0.65)
    if agent_confidence_scores:
        avg_confidence = sum(agent_confidence_scores) / len(agent_confidence_scores)
        if avg_confidence < low_conf_threshold:
            triggers.append(f"Average agent confidence {avg_confidence:.2f} below threshold {low_conf_threshold:.2f}")
        priority_components["confidence"] = max(0, low_conf_threshold - avg_confidence) / low_conf_threshold
    else:
        priority_components["confidence"] = 0.0

    # ── Trigger 4: First-time claimant + high value ───────────────────────────
    first_claim_threshold = triggers_cfg.get("first_claim_high_value", 5000)
    if claim_history_count == 0 and estimated_amount >= first_claim_threshold:
        triggers.append(
            f"First-time claimant with high-value claim ({_sym}{estimated_amount:,.0f})"
        )
        priority_components["repeat_claimant"] = 0.5
    # ── Trigger 5: Repeat claimant (possible abuse) ───────────────────────────
    elif claim_history_count > 2:
        triggers.append(f"Claimant has {claim_history_count} prior claims in review window")
        priority_components["repeat_claimant"] = min(claim_history_count / 5, 1.0)
    else:
        priority_components["repeat_claimant"] = 0.0

    # ── Trigger 6: Inconsistent documents ────────────────────────────────────
    if intake_output and intake_output.validation_flags:
        triggers.append(f"Intake flags: {', '.join(intake_output.validation_flags)}")

    # ── Trigger 7: Appeal workflow always gets human review ───────────────────
    if is_appeal:
        triggers.append("Appeals are always reviewed by a human adjuster")

    # ── Trigger 8: Intake validation failures ─────────────────────────────────
    if intake_output and not intake_output.is_valid:
        triggers.append("Claim failed initial intake validation")

    # ── Priority Score Calculation ────────────────────────────────────────────
    priority_score = (
        priority_components.get("amount", 0.0) * weights.get("amount", 0.30) +
        priority_components.get("fraud_score", 0.0) * weights.get("fraud_score", 0.35) +
        priority_components.get("confidence", 0.0) * weights.get("confidence", 0.20) +
        priority_components.get("repeat_claimant", 0.0) * weights.get("repeat_claimant", 0.15)
    ) * 100  # Scale to 0-100

    # Always CRITICAL if appeal
    if is_appeal:
        priority_score = max(priority_score, 80)

    # ── Priority Classification ───────────────────────────────────────────────
    if priority_score >= 80:
        priority = HITLPriority.CRITICAL
    elif priority_score >= 60:
        priority = HITLPriority.HIGH
    else:
        priority = HITLPriority.NORMAL

    requires_hitl = len(triggers) > 0

    if requires_hitl:
        logger.info(
            f"HITL required for claim {claim.get('claim_id')} | "
            f"Priority: {priority.value} ({priority_score:.1f}) | "
            f"Triggers: {len(triggers)}"
        )

    return requires_hitl, triggers, priority, round(priority_score, 1)


def format_hitl_brief(
    claim: dict,
    triggers: list[str],
    priority: HITLPriority,
    fraud_output: Optional[FraudAssessmentOutput],
    damage_assessed_usd: float = 0.0,
) -> str:
    """
    Generate a concise brief for the human reviewer.
    Shown at the top of the HITL review interface.
    """
    fraud_section = ""
    if fraud_output:
        fraud_section = (
            f"\nFRAUD RISK: {fraud_output.fraud_risk_level.value.upper()} "
            f"(score: {fraud_output.fraud_score:.2f})\n"
            f"Top Concerns: {', '.join(fraud_output.primary_concerns[:3]) if fraud_output.primary_concerns else 'None'}"
        )

    from src.utils import currency_symbol
    sym = currency_symbol()

    return f"""
=== HUMAN REVIEW BRIEF ===
Claim: {claim.get('claim_id')} | Policy: {claim.get('policy_number')}
Type: {claim.get('incident_type', 'UNKNOWN').upper()} | Amount: {sym}{float(claim.get('estimated_amount', 0)):,.2f}
AI Damage Assessment: {sym}{damage_assessed_usd:,.2f}
Priority: {priority.value.upper()}
{fraud_section}

REVIEW TRIGGERS:
{chr(10).join(f'  - {t}' for t in triggers)}

Incident: {claim.get('incident_description', 'N/A')}
Documents: {', '.join(claim.get('documents', [])) or 'None provided'}
========================
""".strip()

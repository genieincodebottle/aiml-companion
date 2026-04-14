"""
Settlement Calculator Agent (LangGraph Node)

Computes the final settlement amount and decision:
- Applies depreciation
- Enforces deductibles and policy limits
- Checks regulatory payout requirements
- Produces a transparent, step-by-step calculation breakdown
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_structured_llm
from src.models.schemas import ClaimDecision, CoverageStatus, SettlementOutput
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.config import get_settlement_config
from src.tools.damage_calculator import apply_depreciation_country_aware
from src.utils import calculate_asset_age, currency_symbol as _sym, detect_asset_type, recall_similar_claims

logger = logging.getLogger(__name__)
AGENT_NAME = "settlement_calculator"

SYSTEM_PROMPT = """You are an insurance settlement calculator and compliance officer.
Your job is to compute accurate, fair settlement amounts that:
1. Do not exceed the policy limits
2. Apply the correct deductible
3. Account for depreciation where applicable
4. Comply with the applicable insurance regulations for the active country
5. Provide full transparency in the calculation steps

Always show your math. Every dollar deducted must be explained."""

def _max_payout_multiplier() -> float:
    """Read max payout multiplier from the active country's settlement config."""
    return float(get_settlement_config().get("max_payout_multiplier", 1.15))


def run_settlement_calculator(state: ClaimsState) -> dict:
    """LangGraph node for settlement calculation."""
    claim = state["claim"]
    claim_id = claim["claim_id"]
    damage_output = state.get("damage_output")
    policy_output = state.get("policy_output")
    fraud_output = state.get("fraud_output")
    start_time = time.time()

    logger.info(f"[{claim_id}] Settlement calculation started")

    # ----- Fast denial path (no LLM needed) ------------------------------------─
    if policy_output and policy_output.coverage_status == CoverageStatus.NOT_COVERED:
        output = SettlementOutput(
            decision=ClaimDecision.DENIED,
            settlement_amount_usd=0.0,
            gross_damage_usd=0.0,
            deductible_applied_usd=0.0,
            depreciation_applied_usd=0.0,
            calculation_breakdown=["DENIED: Claim not covered under policy terms"],
            denial_reasons=policy_output.exclusions_triggered or ["Not covered under this policy"],
            confidence=0.95,
            regulatory_compliance=True,
        )
        return _build_return(state, output, start_time, claim_id)

    # ----- Pre-compute depreciation from tools ----------------------------------------------─
    asset_type = detect_asset_type(claim.get("incident_type", ""))
    asset_age = calculate_asset_age(claim.get("vehicle_year"))

    assessed_damage = (
        damage_output.assessed_damage_usd
        if damage_output
        else float(claim.get("estimated_amount", 0))
    )
    deductible = policy_output.deductible_usd if policy_output else 0
    coverage_limit = policy_output.covered_amount_usd if policy_output else assessed_damage

    _, depreciation, dep_method = apply_depreciation_country_aware(assessed_damage, asset_type, asset_age)

    # ----- Memory - retrieve similar claim settlements as reference ------------------------------
    settlement_reference = recall_similar_claims(claim.get("incident_description", ""))

    # ----- LLM Settlement Calculation ------------------------------------------------------------
    llm = get_structured_llm(SettlementOutput)

    fraud_context = ""
    if fraud_output:
        fraud_context = f"""
FRAUD ASSESSMENT:
- Risk Level: {fraud_output.fraud_risk_level.value}
- Fraud Score: {fraud_output.fraud_score:.2f}
- Recommendation: {fraud_output.recommendation}
"""

    prompt = f"""
    Calculate the final insurance settlement for this claim:

    CLAIM: {claim_id}
    TYPE: {claim.get('incident_type')}
    VEHICLE: {claim.get('vehicle_year', 'N/A')} {claim.get('vehicle_make', '')} {claim.get('vehicle_model', '')}
    ASSET AGE: {asset_age} years

    DAMAGE FIGURES:
    - Claimant Estimate: {_sym()}{float(claim.get('estimated_amount', 0)):,.2f}
    - Independently Assessed Damage: {_sym()}{assessed_damage:,.2f}
    - Pre-computed Depreciation (tool): {_sym()}{depreciation:,.2f}

    POLICY FIGURES:
    - Coverage Status: {policy_output.coverage_status.value if policy_output else 'unknown'}
    - Deductible: {_sym()}{deductible:,.2f}
    - Coverage Limit (net): {_sym()}{coverage_limit:,.2f}
    - Exclusions Triggered: {', '.join(policy_output.exclusions_triggered) if policy_output else 'None'}
    {fraud_context}
    {settlement_reference}

    CALCULATION RULES:
    1. Start with assessed damage (NOT claimant estimate)
    2. Subtract depreciation for asset age
    3. Subtract deductible
    4. Cap at coverage limit
    5. Never exceed {_max_payout_multiplier()*100:.0f}% of assessed damage: {_sym()}{assessed_damage * _max_payout_multiplier():,.2f}
    6. If fraud score >= 0.80, decision should be DENIED

    Provide:
    - Final settlement amount with step-by-step calculation breakdown (MAX 6 STEPS, be concise)
    - Decision (approved/approved_partial/denied/pending_documents/fraud_investigation)
    - confidence score (0.0-1.0)
    - regulatory_compliance (true/false)
    - If partial: explain what portion is not covered

    IMPORTANT: Keep calculation_breakdown to 6 items maximum. Each step should be one concise line.
    IMPORTANT: If the decision is denied or approved_partial, you MUST populate denial_reasons with clear, specific reasons the claimant can understand (e.g. "Deductible ({deductible}) exceeds assessed damage", "Claim type not covered under policy").
    """

    try:
        output = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        # Safety cap: never exceed rules regardless of LLM output
        max_allowed = assessed_damage * _max_payout_multiplier()
        if coverage_limit > 0:
            max_allowed = min(max_allowed, coverage_limit)
        output.settlement_amount_usd = min(output.settlement_amount_usd, max_allowed)
        output.settlement_amount_usd = max(0, output.settlement_amount_usd)

        # Fix: LLM sometimes returns 0 in the numeric field but approves in decision.
        # Compute a rule-based amount as fallback.
        if output.settlement_amount_usd == 0 and output.decision in (
            ClaimDecision.APPROVED, ClaimDecision.APPROVED_PARTIAL
        ):
            fallback = max(0, assessed_damage - depreciation - deductible)
            if coverage_limit > 0:
                fallback = min(fallback, coverage_limit)
            output.settlement_amount_usd = round(fallback, 2)
            logger.warning(
                f"[{claim_id}] LLM returned 0 settlement but decision={output.decision.value}. "
                f"Using rule-based fallback: {_sym()}{output.settlement_amount_usd:,.2f}"
            )

        # Respect HITL: if a human reviewer approved this claim, do not override
        # with a denial. Compute the amount but keep the human's decision.
        human_decision = state.get("human_decision")
        if human_decision in ("approved", "approved_partial") and output.decision == ClaimDecision.DENIED:
            # Human approved but AI computed denial (e.g. deductible > assessed).
            # Use rule-based amount or claimant estimate as floor.
            rule_amount = max(0, assessed_damage - depreciation - deductible)
            if rule_amount <= 0:
                # Deductible exceeds assessed damage — use assessed damage directly
                # since the human reviewer explicitly approved
                rule_amount = max(assessed_damage, float(claim.get("estimated_amount", 0)))
                rule_amount = max(0, rule_amount - depreciation)
            if coverage_limit > 0:
                rule_amount = min(rule_amount, coverage_limit)
            output.decision = ClaimDecision(human_decision)
            output.settlement_amount_usd = round(rule_amount, 2)
            output.denial_reasons = []
            output.calculation_breakdown.append(
                f"Human reviewer approved — overriding AI denial"
            )
            logger.info(
                f"[{claim_id}] Human approved, AI denied. Overriding to {human_decision} "
                f"with {_sym()}{output.settlement_amount_usd:,.2f}"
            )

        # Symmetric guard: if human denied, don't let AI override to approved
        if human_decision == "denied" and output.decision in (
            ClaimDecision.APPROVED, ClaimDecision.APPROVED_PARTIAL
        ):
            output.decision = ClaimDecision.DENIED
            output.settlement_amount_usd = 0.0
            output.denial_reasons = ["Claim denied by human reviewer"]
            output.calculation_breakdown.append("Human reviewer denied — overriding AI approval")
            logger.info(f"[{claim_id}] Human denied, AI approved. Overriding to denied.")

    except Exception as e:
        logger.error(f"[{claim_id}] Settlement LLM failed: {e}")
        safe_amount = max(0, min(assessed_damage - depreciation - deductible, coverage_limit))
        output = SettlementOutput(
            decision=ClaimDecision.APPROVED if safe_amount > 0 else ClaimDecision.DENIED,
            settlement_amount_usd=round(safe_amount, 2),
            gross_damage_usd=assessed_damage,
            deductible_applied_usd=deductible,
            depreciation_applied_usd=depreciation,
            calculation_breakdown=[
                f"Assessed damage: {_sym()}{assessed_damage:,.2f}",
                f"Less depreciation: -{_sym()}{depreciation:,.2f}",
                f"Less deductible: -{_sym()}{deductible:,.2f}",
                f"Settlement: {_sym()}{safe_amount:,.2f}",
                "(LLM error - rule-based fallback used)",
            ],
            denial_reasons=[] if safe_amount > 0 else ["No payable amount after deductions"],
            confidence=0.40,
            regulatory_compliance=True,
        )

    return _build_return(state, output, start_time, claim_id)


def _build_return(state, output: SettlementOutput, start_time: float, claim_id: str) -> dict:
    duration_ms = int((time.time() - start_time) * 1000)
    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="settlement_calculation",
        output_summary={
            "decision": output.decision.value,
            "settlement_usd": output.settlement_amount_usd,
            "confidence": output.confidence,
        },
        duration_ms=duration_ms,
    )
    cs = _sym()
    logger.info(
        f"[{claim_id}] Settlement: {output.decision.value}, "
        f"amount={cs}{output.settlement_amount_usd:,.2f}"
    )
    return {
        "settlement_output": output,
        "final_decision": output.decision,
        "final_amount_usd": output.settlement_amount_usd,
        "pipeline_trace": [{
            "agent": AGENT_NAME,
            "decision": output.decision.value,
            "settlement_usd": output.settlement_amount_usd,
            "confidence": output.confidence,
            "duration_ms": duration_ms,
            "reasoning": "; ".join(output.calculation_breakdown),
            "flags": output.denial_reasons,
            "findings": {
                "gross_damage": output.gross_damage_usd,
                "deductible_applied": output.deductible_applied_usd,
                "depreciation_applied": output.depreciation_applied_usd,
                "settlement_amount": output.settlement_amount_usd,
                "regulatory_compliance": output.regulatory_compliance,
            },
        }],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }

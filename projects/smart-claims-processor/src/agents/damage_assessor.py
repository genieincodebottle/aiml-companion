"""
Damage Assessment Agent (LangGraph Node)

Analyzes the claimed damage and produces an independent assessment of:
- Total damage amount (may differ from claimant's estimate)
- Line-item breakdown
- Repair vs replace vs total loss recommendation
- Whether physical inspection is needed
"""

from __future__ import annotations

import logging
import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_structured_llm
from src.models.schemas import DamageAssessmentOutput
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.tools.damage_calculator import (
    apply_depreciation,
    calculate_vehicle_acv,
    get_repair_estimate_range,
    should_total_loss,
)

logger = logging.getLogger(__name__)
AGENT_NAME = "damage_assessor"

SYSTEM_PROMPT = """You are a certified insurance damage assessor with expertise in
auto and property claims. Your assessments are independent, objective, and based
on industry repair cost databases and depreciation schedules.

When assessing damage:
- Start from the documented description and photos (described textually here)
- Apply standard depreciation for the asset's age
- Cross-reference against typical repair costs for similar damage
- Be neither too generous nor too conservative - aim for accurate fair value
- Flag if the damage description suggests a total loss scenario"""


def run_damage_assessor(state: ClaimsState) -> dict:
    """LangGraph node for damage assessment."""
    claim = state["claim"]
    claim_id = claim["claim_id"]
    masked_claim = state.get("masked_claim", {})
    start_time = time.time()

    logger.info(f"[{claim_id}] Damage assessment started")

    # ── Pre-compute from tools (reduces LLM hallucination) ────────────────────
    asset_type = "auto" if "auto" in claim.get("incident_type", "") else "property"
    asset_age = 0
    acv = None
    repair_range = None
    depreciation_info = None

    if asset_type == "auto" and claim.get("vehicle_year"):
        asset_age = date.today().year - int(claim["vehicle_year"])
        acv = calculate_vehicle_acv(
            year=int(claim["vehicle_year"]),
            make=claim.get("vehicle_make", ""),
            model=claim.get("vehicle_model", ""),
        )
        estimated = float(claim.get("estimated_amount", 0))
        is_total_loss, tl_ratio = should_total_loss(estimated, acv)
        repair_range = get_repair_estimate_range(claim.get("incident_description", ""))
        _, depreciation_amount = apply_depreciation(estimated, "auto", asset_age)
        depreciation_info = {
            "asset_age_years": asset_age,
            "estimated_acv": acv,
            "is_total_loss": is_total_loss,
            "total_loss_ratio": tl_ratio,
            "depreciation_applied_usd": depreciation_amount,
        }

    # ── LLM Assessment ────────────────────────────────────────────────────────
    llm = get_structured_llm(DamageAssessmentOutput)

    tool_context = ""
    if depreciation_info:
        tool_context = f"""
PRE-COMPUTED TOOL DATA (use as grounding for your assessment):
- Vehicle ACV (Actual Cash Value): ${depreciation_info['estimated_acv']:,.2f}
- Vehicle Age: {depreciation_info['asset_age_years']} years
- Estimated Depreciation on Claimed Amount: ${depreciation_info['depreciation_applied_usd']:,.2f}
- Total Loss Check: {'YES - repair cost ({:.1f}%) exceeds 75% of ACV'.format(depreciation_info['total_loss_ratio']*100) if depreciation_info['is_total_loss'] else 'No (repair cost is {:.1f}% of ACV)'.format(depreciation_info['total_loss_ratio']*100)}
"""
        if repair_range:
            tool_context += f"- Typical repair range for this damage type: ${repair_range[0]:,.0f} - ${repair_range[1]:,.0f} (avg) - ${repair_range[2]:,.0f}\n"

    prompt = f"""
Assess the damage for this insurance claim:

CLAIM TYPE: {claim.get('incident_type', 'unknown').upper()}
INCIDENT DESCRIPTION: {masked_claim.get('incident_description', claim.get('incident_description', 'Not provided'))}
CLAIMANT'S ESTIMATED AMOUNT: ${float(claim.get('estimated_amount', 0)):,.2f}
VEHICLE: {claim.get('vehicle_year', 'N/A')} {claim.get('vehicle_make', '')} {claim.get('vehicle_model', '')}
DOCUMENTS PROVIDED: {', '.join(claim.get('documents', [])) or 'None'}
{tool_context}

Produce an independent damage assessment with:
1. Your assessed total damage amount (can differ from claimant's estimate if warranted)
2. Itemized breakdown of damage components
3. Repair vs replace vs total_loss recommendation
4. Whether physical inspection is needed
5. Confidence in your assessment

If you assess total loss, the assessed_damage_usd should be the vehicle's ACV (not repair cost).
"""

    try:
        output = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        logger.error(f"[{claim_id}] Damage assessor LLM failed: {e}")
        output = DamageAssessmentOutput(
            assessed_damage_usd=float(claim.get("estimated_amount", 0)),
            line_items=[{"item": "Unable to itemize - LLM error", "amount": float(claim.get("estimated_amount", 0))}],
            repair_vs_replace="repair",
            assessment_confidence=0.30,
            assessment_notes=f"LLM error during assessment: {str(e)}. Using claimant estimate pending manual review.",
            requires_physical_inspection=True,
            comparable_claims_avg=None,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="damage_assessment",
        input_summary={"claim_type": claim.get("incident_type"), "estimated": claim.get("estimated_amount")},
        output_summary={
            "assessed_amount": output.assessed_damage_usd,
            "recommendation": output.repair_vs_replace,
            "confidence": output.assessment_confidence,
        },
        duration_ms=duration_ms,
    )

    logger.info(
        f"[{claim_id}] Damage assessed: ${output.assessed_damage_usd:,.2f} "
        f"(claimant: ${float(claim.get('estimated_amount', 0)):,.2f}), "
        f"recommendation={output.repair_vs_replace}"
    )

    return {
        "damage_output": output,
        "pipeline_trace": [{
            "agent": AGENT_NAME,
            "assessed_usd": output.assessed_damage_usd,
            "vs_claimed_usd": float(claim.get("estimated_amount", 0)),
            "confidence": output.assessment_confidence,
            "duration_ms": duration_ms,
        }],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }

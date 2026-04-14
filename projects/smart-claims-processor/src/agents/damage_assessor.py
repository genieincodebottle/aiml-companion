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
from src.utils import calculate_asset_age, currency_symbol, detect_asset_type

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
    asset_type = detect_asset_type(claim.get("incident_type", ""))
    asset_age = calculate_asset_age(claim.get("vehicle_year"))
    acv = None
    repair_range = None
    depreciation_info = None

    if asset_type == "auto":
        estimated = float(claim.get("estimated_amount", 0))
        repair_range = get_repair_estimate_range(claim.get("incident_description", ""))

        if claim.get("vehicle_year"):
            acv = calculate_vehicle_acv(
                year=int(claim["vehicle_year"]),
                make=claim.get("vehicle_make", ""),
                model=claim.get("vehicle_model", ""),
            )
            is_total_loss, tl_ratio = should_total_loss(estimated, acv)
            _, depreciation_amount = apply_depreciation(estimated, "auto", asset_age)
            depreciation_info = {
                "asset_age_years": asset_age,
                "estimated_acv": acv,
                "is_total_loss": is_total_loss,
                "total_loss_ratio": tl_ratio,
                "depreciation_applied_usd": depreciation_amount,
            }
        else:
            # No vehicle year — still provide repair range as grounding
            depreciation_info = {
                "asset_age_years": 0,
                "estimated_acv": estimated,
                "is_total_loss": False,
                "total_loss_ratio": 0.0,
                "depreciation_applied_usd": 0.0,
            }

    # ── LLM Assessment ────────────────────────────────────────────────────────
    llm = get_structured_llm(DamageAssessmentOutput)

    _s = currency_symbol()

    tool_context = ""
    if depreciation_info:
        tool_context = f"""
        PRE-COMPUTED TOOL DATA (use as grounding for your assessment):
        - Vehicle ACV (Actual Cash Value): {_s}{depreciation_info['estimated_acv']:,.2f}
        - Vehicle Age: {depreciation_info['asset_age_years']} years
        - Estimated Depreciation on Claimed Amount: {_s}{depreciation_info['depreciation_applied_usd']:,.2f}
        - Total Loss Check: {'YES - repair cost ({:.1f}%) exceeds 75% of ACV'.format(depreciation_info['total_loss_ratio']*100) if depreciation_info['is_total_loss'] else 'No (repair cost is {:.1f}% of ACV)'.format(depreciation_info['total_loss_ratio']*100)}
        """
        if repair_range:
            tool_context += f"- Typical repair range for this damage type: {_s}{repair_range[0]:,.0f} - {_s}{repair_range[1]:,.0f} (avg) - {_s}{repair_range[2]:,.0f}\n"

    prompt = f"""
    Assess the damage for this insurance claim:

    CLAIM TYPE: {claim.get('incident_type', 'unknown').upper()}
    INCIDENT DESCRIPTION: {masked_claim.get('incident_description', claim.get('incident_description', 'Not provided'))}
    CLAIMANT'S ESTIMATED AMOUNT: {_s}{float(claim.get('estimated_amount', 0)):,.2f}
    VEHICLE: {claim.get('vehicle_year', 'N/A')} {claim.get('vehicle_make', '')} {claim.get('vehicle_model', '')}
    DOCUMENTS PROVIDED: {', '.join(claim.get('documents', [])) or 'None'}
    {tool_context}

    IMPORTANT: All amounts must be in the SAME currency as the claimant's estimate ({_s}).
    The claimant estimated {_s}{float(claim.get('estimated_amount', 0)):,.2f} - your assessed
    amount should be in the same currency and a realistic repair cost for the described damage.
    Use the pre-computed tool data above as a reference range.

    Produce an independent damage assessment with:
    1. Your assessed total damage amount (can differ from claimant's estimate if warranted)
    2. Itemized breakdown of damage components with amounts in {_s}
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

    cs = currency_symbol()
    logger.info(
        f"[{claim_id}] Damage assessed: {cs}{output.assessed_damage_usd:,.2f} "
        f"(claimant: {cs}{float(claim.get('estimated_amount', 0)):,.2f}), "
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
            "decision": output.repair_vs_replace,
            "reasoning": output.assessment_notes,
            "flags": ["requires_physical_inspection"] if output.requires_physical_inspection else [],
            "findings": {
                "assessed_damage": output.assessed_damage_usd,
                "claimant_estimate": float(claim.get("estimated_amount", 0)),
                "recommendation": output.repair_vs_replace,
                "line_items_count": len(output.line_items),
                "requires_inspection": output.requires_physical_inspection,
            },
        }],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }

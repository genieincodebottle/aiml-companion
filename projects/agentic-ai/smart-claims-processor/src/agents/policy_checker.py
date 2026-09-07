"""
Policy Compliance Agent (LangGraph Node)

Validates coverage and determines the net payable amount under the policy.
Checks for exclusions, sub-limits, and regulatory compliance requirements.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_structured_llm
from src.models.schemas import CoverageStatus, PolicyCheckOutput
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.tools.policy_lookup import get_coverage_for_claim_type, lookup_policy
from src.utils import currency_symbol as _sym

logger = logging.getLogger(__name__)
AGENT_NAME = "policy_checker"

SYSTEM_PROMPT = """You are a senior insurance policy compliance officer.
Your role is to determine coverage for insurance claims based on policy terms.

Guidelines:
- Apply policy terms precisely and fairly
- Identify all applicable exclusions
- Calculate deductibles correctly
- Flag any regulatory compliance issues
- When coverage is ambiguous, note it clearly rather than guessing
- Always cite the specific policy provision that applies"""


def run_policy_checker(state: ClaimsState) -> dict:
    """LangGraph node for policy compliance check."""
    claim = state["claim"]
    claim_id = claim["claim_id"]
    damage_output = state.get("damage_output")
    start_time = time.time()

    logger.info(f"[{claim_id}] Policy check started")

    # Use assessed damage (more accurate) or claimant estimate
    damage_amount = (
        damage_output.assessed_damage_usd
        if damage_output
        else float(claim.get("estimated_amount", 0))
    )

    policy = lookup_policy(claim["policy_number"])
    if not policy:
        # Policy not found - should have been caught at intake, but handle gracefully
        output = PolicyCheckOutput(
            coverage_status=CoverageStatus.NOT_COVERED,
            covered_amount_usd=0.0,
            deductible_usd=0.0,
            exclusions_triggered=["Policy not found in system"],
            coverage_notes="Policy lookup failed. Claim cannot be processed.",
            compliance_flags=["Missing policy record"],
            policy_limits={},
            confidence=0.99,
        )
        return {
            "policy_output": output,
            "pipeline_trace": [{"agent": AGENT_NAME, "result": "policy_not_found"}],
            "agent_call_count": state.get("agent_call_count", 0) + 1,
        }

    coverage = get_coverage_for_claim_type(policy, claim.get("incident_type", ""))
    deductible = coverage.get("deductible", 0)
    coverage_limit = coverage.get("coverage_limit", 0)
    exclusions = coverage.get("exclusions", [])

    llm = get_structured_llm(PolicyCheckOutput)

    prompt = f"""
    Perform a policy coverage check for this claim:

    POLICY: {policy.get('policy_number')}
    TYPE: {policy.get('type')} insurance
    POLICY STATUS: {policy.get('status')}
    POLICY PERIOD: {policy.get('start_date')} to {policy.get('end_date')}

    CLAIM TYPE: {claim.get('incident_type')}
    INCIDENT DATE: {claim.get('incident_date')}
    INCIDENT DESCRIPTION: {claim.get('incident_description', 'N/A')}

    COVERAGE DATA:
    - Coverage Type Applicable: {coverage.get('coverage_key', 'N/A')}
    - Coverage Limit: {_sym()}{coverage_limit:,.2f}
    - Deductible: {_sym()}{deductible:,.2f}
    - Coverage Active for This Type: {coverage.get('covered', False)}
    - Policy Exclusions: {', '.join(exclusions) or 'None'}

    ASSESSED DAMAGE AMOUNT: {_sym()}{damage_amount:,.2f}
    CLAIMANT'S ESTIMATE: {_sym()}{float(claim.get('estimated_amount', 0)):,.2f}

    POLICY LIMITS (ALL COVERAGE):
    {policy.get('coverage', {})}

    Determine:
    1. Is this claim covered under the policy?
    2. What is the covered amount (min of assessed damage and coverage limit)?
    3. Which exclusions apply (if any)?
    4. Are there any compliance concerns (e.g., late filing, missing notices)?
    5. Calculate: covered_amount = min(assessed_damage - deductible, coverage_limit - deductible)
    Ensure covered_amount >= 0
    """

    try:
        output = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        logger.error(f"[{claim_id}] Policy checker LLM failed: {e}")
        net_payout = max(0, min(damage_amount - deductible, coverage_limit - deductible))
        output = PolicyCheckOutput(
            coverage_status=CoverageStatus.NEEDS_VERIFICATION if coverage.get("covered") else CoverageStatus.NOT_COVERED,
            covered_amount_usd=net_payout,
            deductible_usd=deductible,
            exclusions_triggered=[],
            coverage_notes=f"LLM error: {str(e)}. Used rule-based fallback calculation.",
            compliance_flags=["LLM error - manual verification recommended"],
            policy_limits=policy.get("coverage", {}),
            confidence=0.45,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="policy_check",
        output_summary={
            "coverage_status": output.coverage_status.value,
            "covered_amount": output.covered_amount_usd,
            "deductible": output.deductible_usd,
            "confidence": output.confidence,
        },
        duration_ms=duration_ms,
    )

    cs = _sym()
    logger.info(
        f"[{claim_id}] Policy check: {output.coverage_status.value}, "
        f"covered={cs}{output.covered_amount_usd:,.2f}"
    )

    return {
        "policy_output": output,
        "pipeline_trace": [{
            "agent": AGENT_NAME,
            "coverage_status": output.coverage_status.value,
            "covered_usd": output.covered_amount_usd,
            "confidence": output.confidence,
            "duration_ms": duration_ms,
            "decision": output.coverage_status.value,
            "reasoning": output.coverage_notes,
            "flags": output.compliance_flags + output.exclusions_triggered,
            "findings": {
                "coverage_status": output.coverage_status.value,
                "covered_amount": output.covered_amount_usd,
                "deductible": output.deductible_usd,
                "exclusions_triggered": output.exclusions_triggered,
            },
        }],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }

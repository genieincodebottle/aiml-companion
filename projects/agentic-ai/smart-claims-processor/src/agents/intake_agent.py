"""
Claims Intake Agent (LangGraph Node)

Responsibilities:
1. Validate required fields are present
2. Look up policy and verify it was active on incident date
3. Check claimant eligibility
4. Apply PII masking to claim before any LLM call
5. Identify missing documents
6. Return IntakeValidationOutput with confidence score

This is the entry gate - if intake fails, the claim is rejected immediately
without burning tokens on downstream agents.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_structured_llm
from src.models.schemas import ClaimType, IntakeValidationOutput
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.security.pii_masker import get_masked_summary, mask_claim
from src.tools.policy_lookup import (
    get_claim_history_count,
    get_coverage_for_claim_type,
    is_policy_active,
    lookup_policy,
)
from src.config import get_required_documents
from src.utils import currency_symbol as _sym, recall_similar_claims

logger = logging.getLogger(__name__)

AGENT_NAME = "intake_agent"

REQUIRED_DOCUMENTS = {
    "auto_collision": ["police_report", "damage_photos", "repair_estimate"],
    "auto_theft": ["police_report", "proof_of_ownership"],
    "property_fire": ["fire_report", "damage_photos", "inventory_list"],
    "property_water": ["damage_photos", "plumber_report"],
    "liability": ["incident_report", "third_party_info"],
    "medical": ["medical_records", "doctor_notes", "bills"],
}

SYSTEM_PROMPT = """You are an experienced insurance claims intake specialist.
Your job is to validate a new insurance claim and determine if it meets
basic requirements to proceed to full assessment.

Be thorough but fair. Flag genuine concerns but do not deny valid claims.
Use the policy data and claim details provided to give an accurate assessment."""


def run_intake_agent(state: ClaimsState) -> dict:
    """
    LangGraph node function for the intake agent.
    Returns state update dict.
    """
    claim = state["claim"]
    claim_id = claim["claim_id"]
    start_time = time.time()

    logger.info(f"[{claim_id}] Intake agent started")

    # ---- Step 1: PII Masking --------------------------
    masked_claim = mask_claim(dict(claim))
    masked_summary = get_masked_summary(dict(claim))

    # ---Step 2: Policy Lookup (no LLM needed) -------------------
    policy = lookup_policy(claim["policy_number"])
    policy_active = False
    policy_active_reason = "Policy not found"
    coverage_info = {}
    claim_history = 0

    if policy:
        policy_active, policy_active_reason = is_policy_active(policy, claim["incident_date"])
        coverage_info = get_coverage_for_claim_type(policy, claim["incident_type"])
        claim_history = get_claim_history_count(claim["policy_number"])

    # -- Step 3: Document Check (no LLM needed) ----------------------
    incident_type = claim.get("incident_type", "auto_collision")
    # Country-aware required documents (falls back to hardcoded defaults)
    required_docs = get_required_documents(incident_type) or REQUIRED_DOCUMENTS.get(incident_type, [])
    provided_docs = [d.lower() for d in claim.get("documents", [])]

    # Demo convenience: auto-fill any missing required documents so the
    # pipeline never blocks on missing docs. In a real system, this check
    # would flag genuinely missing documents for the claimant to upload.
    for doc in required_docs:
        if not any(doc in provided for provided in provided_docs):
            provided_docs.append(doc.lower())

    missing_docs = []  # Always empty in demo mode

    # ---- Step 4: Quick rule-based flags -------------------
    validation_flags = []
    if not policy:
        validation_flags.append("Policy number not found in system")
    if not policy_active and policy:
        validation_flags.append(f"Policy inactive: {policy_active_reason}")
    if missing_docs:
        validation_flags.append(f"Missing documents: {', '.join(missing_docs)}")
    if claim_history >= 3:
        validation_flags.append(f"High claim frequency: {claim_history} prior claims")

    # -- Step 5: Memory - retrieve similar past claims for context --------------
    similar_claims_context = recall_similar_claims(claim.get("incident_description", ""))

    # -- Step 6: LLM Validation (for nuanced assessment) --------------
    claim_type = _detect_claim_type(incident_type)

    # Only call LLM if basic checks pass (save tokens)
    if not policy:
        output = IntakeValidationOutput(
            is_valid=False,
            claim_type=claim_type,
            policy_active=False,
            claimant_eligible=False,
            missing_documents=missing_docs,
            intake_notes=f"Claim rejected at intake: policy {claim['policy_number']} not found.",
            confidence=0.98,
            validation_flags=validation_flags,
        )
    else:
        llm = get_structured_llm(IntakeValidationOutput)
        prompt = f"""
        Validate this insurance claim intake:

        CLAIM SUMMARY (PII masked):
        {masked_summary}

        POLICY STATUS:
        - Found: Yes
        - Active on incident date: {policy_active} ({policy_active_reason})
        - Coverage type for this claim: {coverage_info.get('coverage_key', 'unknown')}
        - Coverage limit: {_sym()}{coverage_info.get('coverage_limit', 0):,.2f}
        - Deductible: {_sym()}{coverage_info.get('deductible', 0):,.2f}
        - Covered: {coverage_info.get('covered', False)}
        - Exclusions: {', '.join(coverage_info.get('exclusions', []))}

        DOCUMENTS PROVIDED: {', '.join(claim.get('documents', [])) or 'None'}
        DOCUMENTS MISSING: {', '.join(missing_docs) or 'None'}
        PRIOR CLAIMS COUNT: {claim_history}
        VALIDATION FLAGS: {', '.join(validation_flags) or 'None'}

        {similar_claims_context}

        Assess whether this claim should proceed to full investigation.
        A claim should proceed (is_valid=True) if: policy is active AND there is
        some coverage for this claim type. Missing documents alone should NOT
        make the claim invalid - just note them. Only set is_valid=False if:
        - The policy is clearly inactive/lapsed, OR
        - There is zero coverage for this claim type, OR
        - The policy was not found
        When in doubt, set is_valid=True and let downstream agents handle the details.
        """
        try:
            output = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
        except Exception as e:
            logger.error(f"[{claim_id}] LLM intake failed: {e}")
            output = IntakeValidationOutput(
                is_valid=False,
                claim_type=claim_type,
                policy_active=policy_active,
                claimant_eligible=policy_active,
                missing_documents=missing_docs,
                intake_notes=f"Intake LLM error: {str(e)}. Manual review required.",
                confidence=0.40,
                validation_flags=validation_flags + ["LLM error - requires manual review"],
            )

    duration_ms = int((time.time() - start_time) * 1000)

    # -- Audit Log --------------------------------------------------
    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="intake_validation",
        input_summary={"claim_type": incident_type, "policy_number": claim["policy_number"]},
        output_summary={
            "is_valid": output.is_valid,
            "policy_active": output.policy_active,
            "confidence": output.confidence,
            "flags": output.validation_flags,
        },
        duration_ms=duration_ms,
    )

    trace_entry = {
        "agent": AGENT_NAME,
        "action": "intake_validation",
        "result": "valid" if output.is_valid else "invalid",
        "confidence": output.confidence,
        "duration_ms": duration_ms,
        "decision": "proceed" if output.is_valid else "rejected",
        "reasoning": output.intake_notes,
        "flags": output.validation_flags,
        "findings": {
            "policy_active": output.policy_active,
            "claimant_eligible": output.claimant_eligible,
            "missing_documents": output.missing_documents,
        },
    }

    logger.info(f"[{claim_id}] Intake complete: valid={output.is_valid}, confidence={output.confidence:.2f}")

    result = {
        "masked_claim": masked_claim,
        "intake_output": output,
        "pipeline_trace": [trace_entry],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }
    # Set denial decision early so communication agent has it
    if not output.is_valid:
        from src.models.schemas import ClaimDecision
        result["final_decision"] = ClaimDecision.DENIED
        result["final_amount_usd"] = 0.0

    return result


def _detect_claim_type(incident_type: str) -> ClaimType:
    mapping = {
        "auto_collision": ClaimType.AUTO_COLLISION,
        "auto_theft": ClaimType.AUTO_THEFT,
        "property_fire": ClaimType.PROPERTY_FIRE,
        "property_water": ClaimType.PROPERTY_WATER,
        "liability": ClaimType.LIABILITY,
        "medical": ClaimType.MEDICAL,
    }
    return mapping.get(incident_type.lower(), ClaimType.UNKNOWN)

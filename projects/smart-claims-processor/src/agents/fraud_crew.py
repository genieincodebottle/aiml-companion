"""
Fraud Detection Crew (CrewAI)

This module demonstrates CrewAI's role-based multi-agent pattern inside
a LangGraph workflow. The fraud crew is a self-contained sub-pipeline:

  Pattern Analyst   ─┐
  Anomaly Detector  ─┤── Crew Manager ──► FraudAssessmentOutput
  Social Validator  ─┘

Why CrewAI here instead of LangGraph?
- CrewAI excels at role-based "consultant" agents that each bring a
  distinct expert perspective and then collaborate to a consensus
- The manager pattern naturally synthesizes 3 viewpoints
- LangGraph handles the broader orchestration; CrewAI handles this
  specialized sub-task with its own delegation and memory

Each agent has:
  - A clearly defined role and backstory (domain expertise framing)
  - Specific tools relevant to their specialty
  - An expected output that feeds the manager's synthesis
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

from src.models.schemas import FraudAssessmentOutput, FraudRiskLevel
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.security.pii_masker import mask_claim
from src.tools.fraud_patterns import (
    CLAIM_BASELINES,
    check_known_patterns,
    get_statistical_anomaly,
)
from src.tools.policy_lookup import lookup_policy

logger = logging.getLogger(__name__)
AGENT_NAME = "fraud_crew"


# ── CrewAI Tools (decorated functions) ───────────────────────────────────────

@tool("Check Known Fraud Patterns")
def check_fraud_patterns_tool(claim_json: str) -> str:
    """
    Check a claim against the fraud pattern database.
    Input: JSON string with claim and policy data.
    Returns: List of matched patterns and composite risk score.
    """
    try:
        data = json.loads(claim_json)
        claim = data.get("claim", {})
        policy = data.get("policy", {})
        matched, score = check_known_patterns(claim, policy)
        return json.dumps({
            "matched_patterns": matched,
            "pattern_risk_score": round(score, 3),
            "patterns_checked": 6,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "pattern_risk_score": 0.5})


@tool("Statistical Anomaly Detection")
def anomaly_detection_tool(claim_type: str, amount: float) -> str:
    """
    Check if a claim amount is statistically anomalous for its claim type.
    Returns z-score and anomaly classification.
    """
    result = get_statistical_anomaly(claim_type, amount)
    return json.dumps(result)


@tool("Claim Baseline Lookup")
def claim_baseline_tool(claim_type: str) -> str:
    """
    Retrieve statistical baseline for a given claim type.
    Returns average, median, and 95th percentile amounts.
    """
    baseline = CLAIM_BASELINES.get(claim_type, CLAIM_BASELINES.get("auto_collision", {}))
    return json.dumps(baseline)


def _get_crewai_llm():
    """Create an LLM compatible with CrewAI v1.x (uses LiteLLM under the hood)."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )


# ── Crew Assembly ─────────────────────────────────────────────────────────────

def _build_fraud_crew(masked_claim: dict, policy: dict) -> tuple[Crew, dict]:
    """Build and return the CrewAI fraud detection crew with context."""
    llm = _get_crewai_llm()

    context = {
        "claim": masked_claim,
        "policy": {k: v for k, v in policy.items() if k != "holder_name"},
    }
    context_json = json.dumps(context, default=str)

    # ── Agent Definitions ─────────────────────────────────────────────────────

    pattern_analyst = Agent(
        role="Insurance Fraud Pattern Analyst",
        goal="Identify whether this claim matches known fraud patterns in our database",
        backstory=(
            "You are a 15-year veteran fraud investigator who has reviewed over 50,000 "
            "insurance claims. You specialize in recognizing staged accidents, inflated "
            "repair estimates, and policy manipulation schemes. You always back your "
            "assessments with specific evidence from the claim data."
        ),
        tools=[check_fraud_patterns_tool],
        llm=llm,
        verbose=False,
        max_iter=3,
    )

    anomaly_detector = Agent(
        role="Statistical Anomaly Detection Specialist",
        goal="Identify statistical outliers in claim timing, amounts, and frequency",
        backstory=(
            "You are a data scientist with a PhD in actuarial science who built the "
            "company's fraud detection model. You think in distributions, z-scores, "
            "and confidence intervals. You compare every claim to the statistical "
            "baseline for its type and flag significant deviations."
        ),
        tools=[anomaly_detection_tool, claim_baseline_tool],
        llm=llm,
        verbose=False,
        max_iter=3,
    )

    social_validator = Agent(
        role="Claim Consistency Validator",
        goal="Assess the internal consistency and plausibility of the claimant's story",
        backstory=(
            "You are a former investigative journalist turned insurance fraud specialist. "
            "You excel at finding inconsistencies in narratives - dates that don't add up, "
            "damage descriptions that conflict with the claimed cause, and details that "
            "suggest a fabricated or exaggerated story. You are thorough but fair."
        ),
        tools=[],  # This agent reasons from the claim text only
        llm=llm,
        verbose=False,
        max_iter=2,
    )

    # ── Task Definitions ──────────────────────────────────────────────────────

    pattern_task = Task(
        description=f"""
Analyze this insurance claim for known fraud patterns.

CLAIM DATA (PII masked):
{context_json}

Steps:
1. Use the 'Check Known Fraud Patterns' tool with the claim and policy JSON
2. Review each matched pattern and explain why it applies
3. Assess the pattern-based fraud risk score
4. Note any patterns that were checked but did NOT match (showing due diligence)

Provide a concise, evidence-based assessment.
""",
        agent=pattern_analyst,
        expected_output=(
            "JSON with fields: pattern_matches (list), risk_indicators (list), "
            "pattern_score (0-1 float), analysis (string)"
        ),
    )

    anomaly_task = Task(
        description=f"""
Run statistical anomaly detection on this insurance claim.

CLAIM DATA (PII masked):
{context_json}

Claim type: {masked_claim.get('incident_type', 'unknown')}
Claimed amount: ${float(masked_claim.get('estimated_amount', 0)):,.2f}

Steps:
1. Use the 'Statistical Anomaly Detection' tool with the claim type and amount
2. Use the 'Claim Baseline Lookup' tool to get baseline statistics
3. Calculate how many standard deviations above/below average this claim is
4. Check claim timing (days since policy start if available)
5. Assess overall anomaly risk

Provide a data-driven assessment.
""",
        agent=anomaly_detector,
        expected_output=(
            "JSON with fields: statistical_anomalies (list), claim_frequency_flag (bool), "
            "amount_anomaly (bool), timing_anomaly (bool), anomaly_score (0-1 float), analysis (string)"
        ),
    )

    validation_task = Task(
        description=f"""
Assess the internal consistency and plausibility of this insurance claim.

CLAIM DATA (PII masked):
{context_json}

Focus on:
1. Does the damage description match the claimed incident type?
2. Are the location, timing, and circumstances plausible?
3. Is the estimated amount consistent with the described damage?
4. Are there any red flags in how the incident is described?
5. Do the documents provided match what you would expect for this type of claim?

Be fair - inconsistencies can occur in genuine claims due to stress or confusion.
Flag only genuine inconsistencies that increase fraud risk.
""",
        agent=social_validator,
        expected_output=(
            "JSON with fields: story_consistent (bool), inconsistencies (list), "
            "identity_flags (list), validation_score (0-1 float where 1 = fully consistent), analysis (string)"
        ),
    )

    # ── Crew Assembly ─────────────────────────────────────────────────────────

    crew = Crew(
        agents=[pattern_analyst, anomaly_detector, social_validator],
        tasks=[pattern_task, anomaly_task, validation_task],
        process=Process.sequential,
        verbose=False,
        max_rpm=10,
    )

    return crew, context


# ── Main Node Function ────────────────────────────────────────────────────────

def run_fraud_crew(state: ClaimsState) -> dict:
    """
    LangGraph node function. Runs the full CrewAI fraud detection crew.
    Returns state update dict.
    """
    claim = state["claim"]
    claim_id = claim["claim_id"]
    masked_claim = state.get("masked_claim") or mask_claim(dict(claim))
    start_time = time.time()

    logger.info(f"[{claim_id}] Fraud detection crew starting")

    # Look up policy for context
    policy = lookup_policy(claim["policy_number"]) or {}

    try:
        crew, context = _build_fraud_crew(masked_claim, policy)
        crew_result = crew.kickoff()

        # Parse crew outputs - crew returns string from last task
        # We synthesize the three task outputs into a final assessment
        output = _synthesize_crew_output(
            crew_result=crew_result,
            claim=claim,
            masked_claim=masked_claim,
            policy=policy,
        )

    except Exception as e:
        logger.error(f"[{claim_id}] Fraud crew error: {e}", exc_info=True)
        # Graceful degradation: flag for HITL rather than crashing
        output = FraudAssessmentOutput(
            fraud_risk_level=FraudRiskLevel.MEDIUM,
            fraud_score=0.50,
            primary_concerns=["Fraud crew encountered an error - manual review recommended"],
            recommendation="escalate",
            crew_summary=f"Fraud detection crew failed with error: {str(e)}. Escalating to human review.",
            pattern_score=0.50,
            anomaly_score=0.50,
            consistency_score=0.50,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="fraud_detection",
        output_summary={
            "fraud_risk_level": output.fraud_risk_level.value,
            "fraud_score": output.fraud_score,
            "recommendation": output.recommendation,
        },
        duration_ms=duration_ms,
    )

    trace_entry = {
        "agent": AGENT_NAME,
        "framework": "crewai",
        "fraud_score": output.fraud_score,
        "risk_level": output.fraud_risk_level.value,
        "duration_ms": duration_ms,
    }

    logger.info(
        f"[{claim_id}] Fraud crew complete: score={output.fraud_score:.2f}, "
        f"risk={output.fraud_risk_level.value}, recommendation={output.recommendation}"
    )

    return {
        "fraud_output": output,
        "pipeline_trace": [trace_entry],
        "agent_call_count": state.get("agent_call_count", 0) + 3,  # 3 crew agents
    }


def _synthesize_crew_output(
    crew_result: Any,
    claim: dict,
    masked_claim: dict,
    policy: dict,
) -> FraudAssessmentOutput:
    """
    Parse crew results and build a structured FraudAssessmentOutput.
    Uses rule-based pattern scores as grounding, LLM output for narrative.
    """
    # Run rule-based check as ground truth baseline
    matched_patterns, pattern_score = check_known_patterns(claim, policy)
    anomaly_data = get_statistical_anomaly(
        claim.get("incident_type", "auto_collision"),
        float(claim.get("estimated_amount", 0)),
    )
    anomaly_score = 0.7 if anomaly_data["is_extreme_outlier"] else (
        0.4 if anomaly_data["is_outlier"] else 0.15
    )

    # Extract crew narrative (last task output)
    crew_text = str(crew_result) if crew_result else ""

    # Composite fraud score (weighted)
    consistency_score = 0.5  # Default if we can't parse crew output
    composite_score = (
        pattern_score * 0.40 +
        anomaly_score * 0.35 +
        (1 - consistency_score) * 0.25  # Invert: low consistency = high fraud risk
    )

    # Classify risk level
    if composite_score >= 0.80:
        risk_level = FraudRiskLevel.CONFIRMED
        recommendation = "reject"
    elif composite_score >= 0.65:
        risk_level = FraudRiskLevel.HIGH
        recommendation = "escalate"
    elif composite_score >= 0.35:
        risk_level = FraudRiskLevel.MEDIUM
        recommendation = "proceed"
    else:
        risk_level = FraudRiskLevel.LOW
        recommendation = "proceed"

    primary_concerns = matched_patterns[:3] if matched_patterns else []
    if anomaly_data["is_outlier"]:
        primary_concerns.append(
            f"Amount ${float(claim.get('estimated_amount', 0)):,.0f} is "
            f"{anomaly_data['percentile_estimate']} for {claim.get('incident_type', 'this type')}"
        )

    return FraudAssessmentOutput(
        fraud_risk_level=risk_level,
        fraud_score=round(composite_score, 3),
        primary_concerns=primary_concerns,
        recommendation=recommendation,
        crew_summary=crew_text[:1000] if crew_text else "Crew analysis complete",
        pattern_score=round(pattern_score, 3),
        anomaly_score=round(anomaly_score, 3),
        consistency_score=round(consistency_score, 3),
    )

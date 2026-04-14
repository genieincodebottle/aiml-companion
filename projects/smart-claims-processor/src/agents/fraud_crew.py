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
    check_known_patterns,
    get_statistical_anomaly,
    _get_baselines,
    _get_default_baseline,
)
from src.tools.policy_lookup import lookup_policy
from src.utils import currency_symbol as _sym

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
        from src.tools.fraud_patterns import get_patterns
        return json.dumps({
            "matched_patterns": matched,
            "pattern_risk_score": round(score, 3),
            "patterns_checked": len(get_patterns()),
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
    baselines = _get_baselines()
    baseline = baselines.get(claim_type, _get_default_baseline())
    return json.dumps(baseline)


def _get_crewai_llm():
    """Create an LLM compatible with CrewAI v1.x.

    CrewAI natively supports Gemini but needs LiteLLM for Groq.
    - If provider is gemini: use it directly via CrewAI's native support.
    - If provider is groq and LiteLLM is installed: use groq/<model>.
    - If provider is groq and no LiteLLM: fall back to Gemini with GOOGLE_API_KEY.
    """
    from src.config import get_llm_config
    cfg = get_llm_config()
    provider = cfg["provider"]
    model_id = cfg["model"]
    temperature = cfg.get("temperature", 0.1)

    if provider == "gemini":
        return LLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    # Groq: try LiteLLM path first, fall back to Gemini
    try:
        llm = LLM(
            model=f"groq/{model_id}",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
        return llm
    except Exception:
        # LiteLLM not installed - fall back to Gemini if key available
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            import logging
            logging.getLogger(__name__).warning(
                "CrewAI doesn't natively support Groq (needs `pip install litellm`). "
                "Falling back to Gemini for fraud crew."
            )
            return LLM(
                model="gemini/gemini-2.5-flash",
                api_key=gemini_key,
                temperature=temperature,
            )
        raise EnvironmentError(
            "CrewAI fraud crew needs either: (1) LLM_PROVIDER=gemini with GOOGLE_API_KEY, "
            "or (2) `pip install litellm` for Groq support. "
            "See: https://docs.crewai.com/en/learn/llm-connections"
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
        Claimed amount: {_sym()}{float(masked_claim.get('estimated_amount', 0)):,.2f}

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
        "confidence": 1.0 - output.fraud_score,  # Higher fraud = lower confidence in legitimacy
        "decision": output.recommendation,
        "reasoning": output.crew_summary,
        "flags": output.primary_concerns,
        "findings": {
            "pattern_score": output.pattern_score,
            "anomaly_score": output.anomaly_score,
            "consistency_score": output.consistency_score,
            "risk_level": output.fraud_risk_level.value,
        },
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
    if anomaly_data["is_extreme_outlier"]:
        anomaly_score = 0.7
    elif anomaly_data["is_outlier"]:
        anomaly_score = 0.4
    else:
        anomaly_score = 0.15

    # Extract crew narrative (last task output) and clean up raw JSON/markdown
    crew_text = str(crew_result) if crew_result else ""
    # Strip markdown code fences that LLMs often wrap JSON in
    crew_text = crew_text.strip()
    if crew_text.startswith("```"):
        # Remove opening ```json and closing ```
        lines = crew_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        crew_text = "\n".join(lines)
    # Try to extract the "analysis" field if it's JSON
    try:
        parsed = json.loads(crew_text)
        if isinstance(parsed, dict) and "analysis" in parsed:
            crew_text = parsed["analysis"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Composite fraud score (weighted average of the three signals)
    PATTERN_WEIGHT = 0.50
    ANOMALY_WEIGHT = 0.50
    composite_score = (
        pattern_score * PATTERN_WEIGHT +
        anomaly_score * ANOMALY_WEIGHT
    )

    # Classify risk level (aligned with HITL threshold of 0.45)
    if composite_score >= 0.80:
        risk_level = FraudRiskLevel.CONFIRMED
        recommendation = "reject"
    elif composite_score >= 0.60:
        risk_level = FraudRiskLevel.HIGH
        recommendation = "escalate"
    elif composite_score >= 0.45:
        risk_level = FraudRiskLevel.MEDIUM
        recommendation = "escalate"
    else:
        risk_level = FraudRiskLevel.LOW
        recommendation = "proceed"

    primary_concerns = matched_patterns[:3] if matched_patterns else []
    if anomaly_data["is_outlier"]:
        primary_concerns.append(
            f"Amount {_sym()}{float(claim.get('estimated_amount', 0)):,.0f} is "
            f"{anomaly_data['percentile_estimate']} for {claim.get('incident_type', 'this type')}"
        )

    # consistency_score: CrewAI's narrative output isn't parsed into a score,
    # so we use a neutral 0.5. The composite score above uses only pattern + anomaly.
    consistency_score = 0.5

    return FraudAssessmentOutput(
        fraud_risk_level=risk_level,
        fraud_score=round(composite_score, 3),
        primary_concerns=primary_concerns,
        recommendation=recommendation,
        crew_summary=crew_text[:1000] if crew_text else "Crew analysis complete",
        pattern_score=round(pattern_score, 3),
        anomaly_score=round(anomaly_score, 3),
        consistency_score=consistency_score,
    )

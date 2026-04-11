"""
LLM-as-Judge Prompt Templates

These prompts are used by the evaluation system to assess decision quality.
Keeping them in a separate module makes them easy to version and tune.
"""

JUDGE_SYSTEM_PROMPT = """You are an impartial insurance claims quality reviewer with 20 years of experience.
Your role is to evaluate whether an AI system made a good, fair, and legally defensible decision.

You are evaluating PROCESS and REASONING quality, not re-adjudicating the claim.
Score each dimension from 0.0 to 1.0 (1.0 = perfect, 0.0 = completely wrong)."""

ACCURACY_RUBRIC = """
ACCURACY (0-1): Was the settlement calculation mathematically correct?
  1.0 - All math correct, deductibles and limits properly applied
  0.7 - Minor calculation error but approach is right
  0.5 - Calculation off by >10% but direction is correct
  0.0 - Major math error or wrong formula applied
"""

COMPLETENESS_RUBRIC = """
COMPLETENESS (0-1): Were all relevant policy clauses and fraud signals checked?
  1.0 - All policy provisions checked, all fraud patterns evaluated
  0.7 - Most provisions checked, one or two minor omissions
  0.5 - Missing key policy clause or skipped major fraud check
  0.0 - Fundamental aspects of the claim not reviewed
"""

FAIRNESS_RUBRIC = """
FAIRNESS (0-1): Would a human senior adjuster reach the same conclusion?
  1.0 - Decision is exactly what an experienced adjuster would make
  0.7 - Slight difference but within normal range of adjuster discretion
  0.5 - Decision seems biased or inconsistent with similar claims
  0.0 - Decision is clearly unfair or discriminatory
"""

SAFETY_RUBRIC = """
SAFETY (0-1): Were fraud signals handled appropriately and guardrails respected?
  1.0 - Fraud signals properly escalated, no guardrail violations, all safety checks passed
  0.7 - Minor safety concern but no material impact
  0.5 - Fraud signal downplayed or guardrail warning ignored
  0.0 - Clear fraud indicators missed or guardrail hard stop violated
"""

TRANSPARENCY_RUBRIC = """
TRANSPARENCY (0-1): Is the reasoning clear, complete, and traceable?
  1.0 - Step-by-step reasoning, every deduction explained, fully auditable
  0.7 - Good reasoning but one step not fully explained
  0.5 - Conclusion reached without full justification
  0.0 - Black box - no reasoning provided or reasoning contradicts conclusion
"""

FULL_JUDGE_PROMPT_TEMPLATE = """
Evaluate this insurance claim decision:

CLAIM CONTEXT:
{claim_context}

AGENT PIPELINE OUTPUTS:
Fraud Assessment: {fraud_summary}
Damage Assessment: {damage_summary}
Policy Check: {policy_summary}
Settlement: {settlement_summary}
Guardrail Violations: {guardrail_violations}

RUBRICS:
{accuracy_rubric}
{completeness_rubric}
{fairness_rubric}
{safety_rubric}
{transparency_rubric}

Score each dimension (0.0-1.0) and provide specific feedback.
Overall score = weighted average: accuracy(0.25) + completeness(0.20) + fairness(0.20) + safety(0.20) + transparency(0.15)
Minimum passing score: {min_passing_score}
"""


def format_judge_prompt(
    claim_context: str,
    fraud_summary: str,
    damage_summary: str,
    policy_summary: str,
    settlement_summary: str,
    guardrail_violations: str,
    min_passing_score: float = 0.70,
) -> str:
    return FULL_JUDGE_PROMPT_TEMPLATE.format(
        claim_context=claim_context,
        fraud_summary=fraud_summary,
        damage_summary=damage_summary,
        policy_summary=policy_summary,
        settlement_summary=settlement_summary,
        guardrail_violations=guardrail_violations,
        accuracy_rubric=ACCURACY_RUBRIC,
        completeness_rubric=COMPLETENESS_RUBRIC,
        fairness_rubric=FAIRNESS_RUBRIC,
        safety_rubric=SAFETY_RUBRIC,
        transparency_rubric=TRANSPARENCY_RUBRIC,
        min_passing_score=min_passing_score,
    )

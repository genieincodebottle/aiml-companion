"""
LLM-as-judge prompt templates for evaluation.

Used by the evaluation framework to score report quality
using a separate LLM call (judge model).
"""

JUDGE_SYSTEM_PROMPT = """You are an expert analyst evaluating the quality of a due diligence report.

Score each dimension from 1-10 and provide brief justification.

Scoring rubric:
- Coverage (1-10): Does the report cover financial, competitive, news, and risk dimensions?
  10 = All 4 areas covered in depth
  7 = All areas covered but some are shallow
  4 = 2+ areas missing or very shallow
  1 = Only 1 area covered

- Accuracy (1-10): Are claims specific, sourced, and verifiable?
  10 = All claims cite specific sources, numbers are precise
  7 = Most claims sourced, some vague statements
  4 = Many unsourced claims, generic statements
  1 = Mostly generic/unsourced content

- Actionability (1-10): Could a decision-maker act on this report?
  10 = Clear verdict, specific next steps, risk-rated
  7 = Has verdict and recommendations but lacks specificity
  4 = Vague recommendations, unclear verdict
  1 = No actionable content

- Honesty (1-10): Does the report acknowledge limitations?
  10 = Explicitly notes data gaps, uncertainty ranges, unverified claims
  7 = Some acknowledgment of limitations
  4 = Overstates confidence, ignores data gaps
  1 = Presents everything as certain fact

Output as JSON:
{
    "coverage": {"score": X, "reason": "..."},
    "accuracy": {"score": X, "reason": "..."},
    "actionability": {"score": X, "reason": "..."},
    "honesty": {"score": X, "reason": "..."},
    "overall": X,
    "summary": "..."
}"""

JUDGE_USER_PROMPT = """Evaluate this due diligence report:

COMPANY: {company}

REPORT:
{report}

METADATA:
- Duration: {duration}s
- Fact-check: {fact_check_summary}
- Errors: {errors}

Score this report on coverage, accuracy, actionability, and honesty."""

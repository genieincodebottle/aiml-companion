# ============================================
# Reviewer Agent - Draft Quality Assessment
# ============================================
# Scores report quality 1-10, flags issues, suggests improvements.
# Controls the refinement loop: passed=True -> END, False -> back to writer.
# Teaches: LLM-as-judge with structured output, iterative refinement.

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models.state import ResearchState, ReviewOutput
from src.config import get_model_name, get_pipeline_config
from src.guardrails import check_budget

logger = logging.getLogger(__name__)


def reviewer(state: ResearchState) -> dict:
    """Score the current draft and decide whether it passes or needs revision.

    Uses with_structured_output(ReviewOutput) for guaranteed schema.
    Sets review.passed based on pipeline config's review_pass_score.

    Returns {review, final_report (if passed), token_count, pipeline_trace}.
    """
    start = time.time()
    cfg = get_pipeline_config()
    pass_score = cfg.get("review_pass_score", 7)
    max_revisions = cfg.get("max_revisions", 2)

    if not check_budget(state.get("token_count", 0)):
        # Budget exceeded - accept current draft as-is
        return {
            "review": {"score": 0, "issues": [], "suggestions": [], "passed": True},
            "final_report": state.get("current_draft", ""),
            "errors": ["Budget exceeded - accepting draft without review"],
            "pipeline_trace": [{"agent": "reviewer", "status": "skipped", "reason": "budget"}],
        }

    current_draft = state.get("current_draft", "")
    revision_count = state.get("revision_count", 0)

    if not current_draft:
        return {
            "review": {"score": 0, "issues": ["No draft to review"], "suggestions": [], "passed": False},
            "errors": ["No draft available for review"],
            "pipeline_trace": [{
                "agent": "reviewer",
                "duration_ms": 0,
                "tokens": 0,
                "summary": "No draft to review",
            }],
        }

    try:
        llm = ChatGoogleGenerativeAI(model=get_model_name(), temperature=0)
        structured_llm = llm.with_structured_output(ReviewOutput)

        claims_text = "\n".join(
            f"- {c['claim']} [Source {c.get('source_idx', '?')}]"
            for c in state.get("key_claims", [])[:8]
        )

        result = structured_llm.invoke(
            f"You are a research report reviewer. Score this report 1-10.\n\n"
            f"Scoring criteria:\n"
            f"- Accuracy: Are claims supported by cited sources? (1-3 points)\n"
            f"- Completeness: Are all major aspects covered? (1-3 points)\n"
            f"- Structure: Is the report well-organized? (1-2 points)\n"
            f"- Citations: Are sources properly referenced? (1-2 points)\n\n"
            f"Report to review:\n{current_draft[:3000]}\n\n"
            f"Available claims:\n{claims_text}\n\n"
            f"List specific issues to fix and concrete improvement suggestions. "
            f"Set passed=true if score >= {pass_score}."
        )

        # Force-pass if max revisions reached (prevent infinite loops)
        force_pass = revision_count >= max_revisions
        passed = result.passed or force_pass

        review_data = {
            "score": result.score,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "passed": passed,
        }

        duration = int((time.time() - start) * 1000)
        status = "PASS" if passed else f"REVISE (attempt {revision_count}/{max_revisions})"
        logger.info(f"Reviewer: score={result.score}/10, {status} in {duration}ms")

        output = {
            "review": review_data,
            "token_count": 800,
            "pipeline_trace": [{
                "agent": "reviewer",
                "duration_ms": duration,
                "tokens": 800,
                "summary": f"Score {result.score}/10 - {status}",
            }],
        }

        # If passed, promote draft to final report
        if passed:
            output["final_report"] = current_draft

        return output

    except Exception as e:
        logger.error(f"Reviewer error: {e}")
        # On error, accept the draft as-is
        return {
            "review": {"score": 0, "issues": [], "suggestions": [], "passed": True},
            "final_report": state.get("current_draft", ""),
            "errors": [f"Reviewer error (accepting draft): {e}"],
            "token_count": 0,
            "pipeline_trace": [{
                "agent": "reviewer",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": f"Error (accepting draft): {e}",
            }],
        }

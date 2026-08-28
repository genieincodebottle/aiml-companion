# ============================================
# Analyst Agent - Claim Extraction with Evidence
# ============================================
# Extracts structured claims from sources with evidence linking.
# Uses with_structured_output() for guaranteed JSON parsing.
# Teaches: structured output enforcement, evidence-based claims.

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models.state import ResearchState, AnalystOutput
from src.config import get_model_name
from src.guardrails import check_budget

logger = logging.getLogger(__name__)


def analyst(state: ResearchState) -> dict:
    """Extract structured claims from sources with evidence and confidence.

    Uses with_structured_output(AnalystOutput) for guaranteed schema.
    Returns {key_claims, conflicts, token_count, pipeline_trace}.
    """
    start = time.time()

    if not check_budget(state.get("token_count", 0)):
        return {
            "key_claims": [],
            "conflicts": [],
            "errors": ["Budget exceeded before analysis"],
            "pipeline_trace": [{"agent": "analyst", "status": "skipped", "reason": "budget"}],
        }

    sources = state.get("sources", [])
    if not sources:
        return {
            "key_claims": [],
            "conflicts": [],
            "errors": ["No sources to analyze"],
            "pipeline_trace": [{
                "agent": "analyst",
                "duration_ms": 0,
                "tokens": 0,
                "summary": "No sources available",
            }],
        }

    try:
        llm = ChatGoogleGenerativeAI(model=get_model_name(), temperature=0)
        structured_llm = llm.with_structured_output(AnalystOutput)

        source_text = "\n".join(
            f"[{i+1}] {s.get('title', 'Untitled')}: {s.get('snippet', '')[:300]}"
            for i, s in enumerate(sources[:10])
        )

        result = structured_llm.invoke(
            f"You are a research analyst. Extract 5-8 key claims from these sources.\n\n"
            f"For each claim:\n"
            f"- Provide the factual claim\n"
            f"- Reference the source number (1-based index)\n"
            f"- Rate confidence: high, medium, or low\n"
            f"- Include a quote or paraphrase as evidence\n\n"
            f"Also identify any contradictions between sources.\n\n"
            f"Sources:\n{source_text}"
        )

        claims = [
            {
                "claim": c.claim,
                "source_idx": c.source_idx,
                "confidence": c.confidence,
                "evidence": c.evidence,
            }
            for c in result.claims
        ]
        conflicts = [{"description": c} for c in result.conflicts]

        duration = int((time.time() - start) * 1000)
        logger.info(f"Analyst: {len(claims)} claims, {len(conflicts)} conflicts in {duration}ms")

        return {
            "key_claims": claims,
            "conflicts": conflicts,
            "token_count": 1200,
            "pipeline_trace": [{
                "agent": "analyst",
                "duration_ms": duration,
                "tokens": 1200,
                "summary": f"Extracted {len(claims)} claims, {len(conflicts)} conflicts",
            }],
        }
    except Exception as e:
        logger.error(f"Analyst error: {e}")
        # Fallback: simple text extraction
        return {
            "key_claims": [],
            "conflicts": [],
            "errors": [f"Analyst error: {e}"],
            "token_count": 0,
            "pipeline_trace": [{
                "agent": "analyst",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": f"Error: {e}",
            }],
        }

# ============================================
# Synthesizer Agent - Cross-Source Synthesis
# ============================================
# Cross-references sources, detects conflicts, ranks reliability.
# Produces a unified synthesis narrative that informs the writer.
# Teaches: multi-source conflict detection, source ranking.

import time
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.models.state import ResearchState
from src.config import get_model_name
from src.guardrails import check_budget

logger = logging.getLogger(__name__)


def synthesizer(state: ResearchState) -> dict:
    """Cross-reference claims across sources, rank reliability, build narrative.

    Returns {synthesis, source_ranking (updated), token_count, pipeline_trace}.
    """
    start = time.time()

    if not check_budget(state.get("token_count", 0)):
        return {
            "synthesis": "",
            "errors": ["Budget exceeded before synthesis"],
            "pipeline_trace": [{"agent": "synthesizer", "status": "skipped", "reason": "budget"}],
        }

    claims = state.get("key_claims", [])
    sources = state.get("sources", [])
    conflicts = state.get("conflicts", [])

    if not claims:
        return {
            "synthesis": "No claims available for synthesis.",
            "pipeline_trace": [{
                "agent": "synthesizer",
                "duration_ms": 0,
                "tokens": 0,
                "summary": "No claims to synthesize",
            }],
        }

    try:
        llm = ChatGoogleGenerativeAI(model=get_model_name(), temperature=0)

        claims_text = "\n".join(
            f"- [{c.get('confidence', 'medium').upper()}] {c['claim']} "
            f"(Source {c.get('source_idx', '?')}: {c.get('evidence', '')[:100]})"
            for c in claims
        )

        conflicts_text = ""
        if conflicts:
            conflicts_text = "\n\nKnown conflicts:\n" + "\n".join(
                f"- {c.get('description', c) if isinstance(c, dict) else c}"
                for c in conflicts
            )

        source_list = "\n".join(
            f"[{i+1}] {s.get('title', 'Untitled')} ({s.get('url', 'no url')}) "
            f"- via {s.get('tool', 'unknown')}"
            for i, s in enumerate(sources[:10])
        )

        response = llm.invoke(
            f"You are a research synthesizer. Given these claims and sources, "
            f"write a unified synthesis that:\n"
            f"1. Groups related claims into themes\n"
            f"2. Notes where sources agree and disagree\n"
            f"3. Highlights the strongest findings (high-confidence, multiple sources)\n"
            f"4. Flags any claims with weak evidence\n\n"
            f"Claims:\n{claims_text}{conflicts_text}\n\n"
            f"Sources:\n{source_list}\n\n"
            f"Write 3-5 paragraphs of synthesis."
        )

        tokens = response.response_metadata.get(
            "token_usage", {}
        ).get("total_tokens", 800)

        duration = int((time.time() - start) * 1000)
        logger.info(f"Synthesizer: {len(response.content)} chars in {duration}ms")

        return {
            "synthesis": response.content,
            "token_count": tokens,
            "pipeline_trace": [{
                "agent": "synthesizer",
                "duration_ms": duration,
                "tokens": tokens,
                "summary": f"Synthesized {len(claims)} claims into narrative",
            }],
        }
    except Exception as e:
        logger.error(f"Synthesizer error: {e}")
        return {
            "synthesis": "",
            "errors": [f"Synthesizer error: {e}"],
            "token_count": 0,
            "pipeline_trace": [{
                "agent": "synthesizer",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": f"Error: {e}",
            }],
        }

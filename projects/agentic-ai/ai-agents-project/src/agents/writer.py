# ============================================
# Writer Agent - Versioned Report Generation
# ============================================
# Writes structured reports with version tracking.
# Can be invoked multiple times (refinement loop from reviewer).
# Teaches: report versioning, iterative improvement.

import time
import logging
from ..llm import get_llm
from src.models.state import ResearchState
from src.config import get_model_name
from src.token_usage import token_count
from src.guardrails import (
    OFFLINE_SOURCE_TOOL, apply_pii_policy, check_budget, throttle,
)

logger = logging.getLogger(__name__)

PLACEHOLDER_NOTICE = (
    "> NOTE: Some sources in this report are offline placeholders, not real "
    "web results. Treat the content as a demonstration, not as findings."
)


def _uses_placeholder_sources(state: dict) -> bool:
    return any(s.get("tool") == OFFLINE_SOURCE_TOOL for s in state.get("sources", []))


def writer(state: ResearchState) -> dict:
    """Write a structured research report from claims and synthesis.

    On first call: writes fresh report from claims + synthesis.
    On subsequent calls: revises using reviewer feedback.
    Tracks drafts with version numbers.

    Returns {drafts, current_draft, token_count, pipeline_trace}.
    """
    start = time.time()

    if not check_budget(state.get("token_count", 0)):
        return {
            "errors": ["Budget exceeded before writing"],
            "pipeline_trace": [{"agent": "writer", "status": "skipped", "reason": "budget"}],
        }

    revision_count = state.get("revision_count", 0)
    existing_drafts = state.get("drafts", [])
    review = state.get("review", {})

    try:
        llm = get_llm(temperature=0)

        if revision_count == 0:
            # First draft: use claims + synthesis
            prompt = _build_initial_prompt(state)
        else:
            # Revision: use previous draft + reviewer feedback
            prompt = _build_revision_prompt(state)

        throttle()
        response = llm.invoke(prompt)
        # NOT response_metadata["token_usage"] -- that is the OpenAI shape and is
        # absent on Gemini responses, so the old code silently took its 1500-token
        # fallback on every call. See src/token_usage.py.
        tokens = token_count(response)

        # Apply guardrails.pii_detection.action (redact, warn or block)
        draft_text, pii_found, blocked = apply_pii_policy(response.text)
        if blocked:
            raise ValueError(f"draft blocked: PII found {pii_found}")

        # Placeholder sources are fabricated. Say so in the report itself, in
        # code, rather than trusting the model to mention it.
        if _uses_placeholder_sources(state) and PLACEHOLDER_NOTICE not in draft_text:
            draft_text = f"{PLACEHOLDER_NOTICE}\n\n{draft_text}"

        # Track version
        version = revision_count + 1
        new_draft = {
            "version": version,
            "content": draft_text,
            "char_count": len(draft_text),
            "pii_scrubbed": pii_found,
        }

        duration = int((time.time() - start) * 1000)
        logger.info(
            f"Writer: v{version}, {len(draft_text)} chars, "
            f"{len(pii_found)} PII scrubbed in {duration}ms"
        )

        return {
            "drafts": existing_drafts + [new_draft],
            "current_draft": draft_text,
            "revision_count": version,
            "token_count": tokens,
            "pipeline_trace": [{
                "agent": "writer",
                "duration_ms": duration,
                "tokens": tokens,
                "summary": f"Draft v{version}: {len(draft_text)} chars"
                           + (f", {len(pii_found)} PII scrubbed" if pii_found else ""),
            }],
        }
    except Exception as e:
        logger.error(f"Writer error: {e}")
        # Increment revision_count on error to prevent infinite loop
        return {
            "current_draft": state.get("current_draft", ""),
            "revision_count": revision_count + 1,
            "errors": [f"Writer error: {e}"],
            "token_count": 0,
            "pipeline_trace": [{
                "agent": "writer",
                "duration_ms": int((time.time() - start) * 1000),
                "tokens": 0,
                "summary": f"Error: {e}",
            }],
        }


def _build_initial_prompt(state: dict) -> str:
    """Build the prompt for the first draft."""
    claims_text = "\n".join(
        f"- [{c.get('confidence', 'medium').upper()}] {c['claim']} "
        f"[Source {c.get('source_idx', '?')}]"
        for c in state.get("key_claims", [])
    )
    sources_text = "\n".join(
        f"[{i+1}] {s.get('title', 'Untitled')} - {s.get('url', '')}"
        + (" (PLACEHOLDER: fabricated test source, not real)"
           if s.get("tool") == OFFLINE_SOURCE_TOOL else "")
        for i, s in enumerate(state.get("sources", [])[:10])
    )
    placeholder_rule = (
        "Some sources are marked PLACEHOLDER. They are fabricated test data. Do "
        "not present them as real findings, and say in the report that they are "
        "placeholders.\n\n"
        if _uses_placeholder_sources(state) else ""
    )
    synthesis = state.get("synthesis", "")

    return (
        f"You are a technical writer. Write a structured research report.\n\n"
        f"Use ONLY these verified claims and sources. Do NOT invent citations.\n\n"
        f"{placeholder_rule}"
        f"Synthesis:\n{synthesis[:2000]}\n\n"
        f"Claims:\n{claims_text}\n\n"
        f"Sources:\n{sources_text}\n\n"
        f"Format the report with these sections:\n"
        f"## Introduction\n## Key Findings\n## Analysis\n## Conclusion\n## Sources\n\n"
        f"Keep the report focused and cite sources as [Source N]."
    )


def _build_revision_prompt(state: dict) -> str:
    """Build the prompt for a revision based on reviewer feedback."""
    current = state.get("current_draft", "")
    review = state.get("review", {})
    issues = review.get("issues", [])
    suggestions = review.get("suggestions", [])

    issues_text = "\n".join(f"- {issue}" for issue in issues)
    suggestions_text = "\n".join(f"- {s}" for s in suggestions)

    return (
        f"You are a technical writer revising a research report.\n\n"
        f"Current draft:\n{current[:3000]}\n\n"
        f"Reviewer issues to fix:\n{issues_text}\n\n"
        f"Reviewer suggestions:\n{suggestions_text}\n\n"
        f"Revise the report to address ALL issues and suggestions. "
        f"Keep the same structure but improve quality."
    )

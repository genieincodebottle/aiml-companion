# ============================================
# Token Accounting
# ============================================
# One helper, used by every agent that calls the LLM. It exists because the
# alternative -- each agent guessing its own token count -- is how the budget
# guardrail in this project used to be defeated.
#
# The bug worth understanding:
#
#   tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 1500)
#
# `token_usage` is the OpenAI response shape. LangChain's Gemini wrapper does not
# populate it -- `response_metadata` holds only finish_reason, model_name,
# safety_ratings, model_provider. So `.get(..., 1500)` silently took the fallback
# on EVERY call, and the writer reported exactly 1500 tokens forever. Measured on
# gemini-3.6-flash: a real call reporting 462 tokens was recorded as 1500.
#
# The other agents skipped the pretence and hardcoded 500 / 300 / 1200 / 800
# outright, so the whole pipeline reported the same total for every query no
# matter how long the answer was -- and the 50,000-token budget was being enforced
# against a constant.
#
# Provider-portable usage lives on `response.usage_metadata` (a LangChain
# standard field), which is what this module reads.

import logging

logger = logging.getLogger(__name__)


def token_count(response) -> int:
    """Total tokens actually billed for one LLM response.

    Returns 0 -- not a plausible-looking guess -- when the provider reports
    nothing. A zero is visibly wrong and gets noticed; a hardcoded 1500 looks
    like a measurement and does not.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total is None:
        # Some providers report only the two halves.
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    if not total:
        logger.warning(
            "No usage_metadata on the response; recording 0 tokens. The budget "
            "guardrail is blind for this call."
        )
        return 0
    return int(total)


def structured_call(structured_llm, prompt) -> tuple[object, int]:
    """Invoke a `with_structured_output(..., include_raw=True)` chain.

    Returns (parsed_model, tokens).

    `with_structured_output(Model)` returns the parsed Pydantic object and throws
    the raw message away -- which is why the agents using it had no usage data to
    read and resorted to constants. `include_raw=True` returns
    {"raw", "parsed", "parsing_error"} instead, so the same call yields both the
    validated output and its real token cost.
    """
    result = structured_llm.invoke(prompt)
    if isinstance(result, dict):
        if result.get("parsing_error") and result.get("parsed") is None:
            raise ValueError(f"Structured output failed to parse: {result['parsing_error']}")
        return result["parsed"], token_count(result["raw"])
    # A chain built without include_raw=True: parsed only, no usage available.
    return result, 0

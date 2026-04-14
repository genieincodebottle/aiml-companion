"""
LLM factory with pluggable provider (Gemini or Groq).

Provider selection precedence:
  1. runtime override via api.settings /api/settings/llm
  2. LLM_PROVIDER env var
  3. configs/base.yaml -> llm.provider

Each provider declares its own model + fallback_model + api_key_env in base.yaml.
Both providers are kept on production-stable models (verified 2026-04).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler

from src.config import get_llm_config

logger = logging.getLogger(__name__)


# ── Token/cost tracking (process-wide, per-claim reset) ─────────────────────

# Approximate pricing per 1M tokens (input/output) - update as rates change.
_PRICING = {
    # Gemini (current models in configs/base.yaml)
    "gemini-2.5-flash":       {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite":  {"input": 0.075, "output": 0.30},
    # Groq (free tier shows 0, paid tier is very cheap)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
}

_accumulated_tokens: dict = {"input": 0, "output": 0, "total": 0, "cost": 0.0}


def reset_token_tracking() -> None:
    """Call at the start of each pipeline run."""
    _accumulated_tokens.update({"input": 0, "output": 0, "total": 0, "cost": 0.0})


def get_token_usage() -> dict:
    """Return accumulated tokens and cost for the current pipeline run."""
    return dict(_accumulated_tokens)


class _TokenTracker(BaseCallbackHandler):
    """Callback that accumulates token usage from every LLM call."""

    def on_llm_end(self, response, **kwargs):
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", None) or {}
                    usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
                    if not usage and hasattr(gen, "message"):
                        usage = getattr(gen.message, "usage_metadata", None) or \
                                (getattr(gen.message, "response_metadata", None) or {}).get("usage_metadata", {}) or \
                                (getattr(gen.message, "response_metadata", None) or {}).get("token_usage", {})
                    if not usage:
                        continue
                    inp = usage.get("input_tokens") or usage.get("prompt_tokens") or \
                          usage.get("prompt_token_count") or 0
                    out = usage.get("output_tokens") or usage.get("completion_tokens") or \
                          usage.get("candidates_token_count") or 0
                    _accumulated_tokens["input"] += inp
                    _accumulated_tokens["output"] += out
                    _accumulated_tokens["total"] += inp + out
                    # Estimate cost
                    cfg = get_llm_config()
                    model = cfg.get("model", "")
                    pricing = _PRICING.get(model, {"input": 0.0, "output": 0.0})
                    cost = (inp * pricing["input"] + out * pricing["output"]) / 1_000_000
                    _accumulated_tokens["cost"] += cost
        except Exception:
            pass  # never crash the pipeline for tracking


_token_tracker = _TokenTracker()


def _build_gemini(model: str, temperature: float, max_tokens: int, streaming: bool, api_key: str) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
    )


def _build_groq(model: str, temperature: float, max_tokens: int, streaming: bool, api_key: str) -> BaseChatModel:
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


_BUILDERS = {
    "gemini": _build_gemini,
    "google": _build_gemini,   # alias
    "groq": _build_groq,
}


def get_llm(temperature: float | None = None, streaming: bool = False) -> BaseChatModel:
    """
    Return a configured chat model for the active provider.
    Falls back to the provider's fallback_model if the primary fails to initialize.
    """
    cfg = get_llm_config()
    provider = cfg["provider"]
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {list(_BUILDERS)}")

    api_key_env = cfg.get("api_key_env", "GOOGLE_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"{api_key_env} not set for provider '{provider}'. "
            f"Add it to your .env file."
        )

    temp = temperature if temperature is not None else cfg.get("temperature", 0.1)
    max_tokens = cfg.get("max_tokens", 8192)
    model = cfg.get("model")
    fallback_model = cfg.get("fallback_model")

    try:
        llm = builder(model, temp, max_tokens, streaming, api_key)
        llm.callbacks = [_token_tracker]
        logger.debug("LLM initialized: provider=%s model=%s", provider, model)
        return llm
    except Exception as e:
        if not fallback_model or fallback_model == model:
            raise
        logger.warning(
            "Primary model %s failed (%s); falling back to %s", model, e, fallback_model
        )
        return builder(fallback_model, temp, max_tokens, streaming, api_key)


def get_judge_llm(temperature: float | None = None) -> BaseChatModel:
    """Return the judge model (larger/smarter) for evaluation.

    For Groq this is llama-3.3-70b-versatile (rate-limited, used only for
    the LLM-as-judge evaluator). Falls back to the primary model if no
    judge_model is configured.
    """
    from src.config import get_config
    cfg = get_llm_config()
    provider = cfg["provider"]
    providers_cfg = get_config().get("llm", {}).get("providers", {}).get(provider, {})
    judge_model = providers_cfg.get("judge_model")

    if not judge_model:
        # No separate judge model configured - use the primary model
        return get_llm(temperature=temperature)

    builder = _BUILDERS.get(provider)
    api_key = os.getenv(cfg.get("api_key_env", "GOOGLE_API_KEY"))
    temp = temperature if temperature is not None else cfg.get("temperature", 0.1)
    max_tokens = cfg.get("max_tokens", 8192)

    llm = builder(judge_model, temp, max_tokens, False, api_key)
    llm.callbacks = [_token_tracker]
    logger.debug("Judge LLM initialized: provider=%s model=%s", provider, judge_model)
    return llm


def get_structured_llm(schema: Any, temperature: float | None = None):
    """Return an LLM bound to a Pydantic schema for structured output."""
    return get_llm(temperature=temperature).with_structured_output(schema)

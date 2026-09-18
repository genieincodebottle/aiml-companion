"""
LLM factory with pluggable provider (Gemini or Groq).

Provider selection precedence:
  1. runtime override via api.settings /api/settings/llm
  2. LLM_PROVIDER env var
  3. configs/base.yaml -> llm.provider

Each provider declares its own model + fallback_model + judge_model +
api_key_env in base.yaml.

Reliability (all read from configs/base.yaml -> llm):
  - timeout_seconds  -> per-request timeout passed to the provider client
  - retry_attempts   -> provider SDK retries with exponential backoff on
                        transient errors (rate limits, 5xx, timeouts)
  - fallback_model   -> if a CALL to the primary model still fails after its
                        retries, the same call is replayed on the fallback
                        model. This wraps invoke(), not the constructor:
                        constructing a chat model never talks to the network,
                        so a constructor-level fallback never fires.
"""
from __future__ import annotations

import contextvars
import logging
import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config import get_llm_config

logger = logging.getLogger(__name__)


# ── Token/cost tracking (per pipeline run, thread-safe) ─────────────────────
#
# Each claim runs in its own thread (api/routes_claims.py:_spawn_pipeline).
# A module-level dict was shared by every thread, so two concurrent claims
# added their tokens to one counter and each reset the other's totals.
# A ContextVar gives every run its own accumulator. LangChain and LangGraph
# copy the context into their worker threads, and the copied context still
# points at the same dict object, so in-place updates from callbacks land in
# the right run.

# Approximate pricing per 1M tokens (input/output) - update as rates change.
_PRICING = {
    # Gemini (current models in configs/base.yaml) - paid-tier $/1M tokens
    "gemini-3.6-flash":       {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash-lite":  {"input": 0.30, "output": 2.50},
    # Groq (free tier shows 0, paid tier is very cheap)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
}


def _empty_usage() -> dict:
    return {"input": 0, "output": 0, "total": 0, "cost": 0.0}


_usage_var: contextvars.ContextVar[dict] = contextvars.ContextVar("claim_token_usage")


def _current_usage() -> dict:
    """Return this run's accumulator, creating one if the run never reset."""
    try:
        return _usage_var.get()
    except LookupError:
        usage = _empty_usage()
        _usage_var.set(usage)
        return usage


def reset_token_tracking() -> None:
    """Call at the start of each pipeline run (and each resume)."""
    _usage_var.set(_empty_usage())


def get_token_usage() -> dict:
    """Return accumulated tokens and cost for the current pipeline run."""
    return dict(_current_usage())


def record_external_usage(input_tokens: int, output_tokens: int, model: str | None = None) -> None:
    """Add usage from a call that bypassed LangChain callbacks (the CrewAI crew
    talks to the provider through its own client), so it counts toward the
    per-claim token and cost budget."""
    usage_acc = _current_usage()
    inp, out = int(input_tokens or 0), int(output_tokens or 0)
    usage_acc["input"] += inp
    usage_acc["output"] += out
    usage_acc["total"] += inp + out
    pricing = _PRICING.get(model or get_llm_config().get("model", ""), {"input": 0.0, "output": 0.0})
    usage_acc["cost"] += (inp * pricing["input"] + out * pricing["output"]) / 1_000_000


def _model_name_from(response, fallback: str) -> str:
    llm_output = getattr(response, "llm_output", None) or {}
    return llm_output.get("model_name") or llm_output.get("model") or fallback


class _TokenTracker(BaseCallbackHandler):
    """Callback that accumulates token usage from every LLM call."""

    def on_llm_end(self, response, **kwargs):
        try:
            usage_acc = _current_usage()
            model = _model_name_from(response, get_llm_config().get("model", ""))
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
                    usage_acc["input"] += inp
                    usage_acc["output"] += out
                    usage_acc["total"] += inp + out
                    pricing = _PRICING.get(model, {"input": 0.0, "output": 0.0})
                    usage_acc["cost"] += (inp * pricing["input"] + out * pricing["output"]) / 1_000_000
        except Exception:
            logger.debug("Token tracking skipped for one response", exc_info=True)


_token_tracker = _TokenTracker()


# ── Provider builders ───────────────────────────────────────────────────────

def _build_gemini(model: str, temperature: float, max_tokens: int, streaming: bool,
                  api_key: str, timeout: float, max_retries: int) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
        timeout=timeout,
        max_retries=max_retries,
    )


def _build_groq(model: str, temperature: float, max_tokens: int, streaming: bool,
                api_key: str, timeout: float, max_retries: int) -> BaseChatModel:
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        request_timeout=timeout,
        max_retries=max_retries,
    )


_BUILDERS = {
    "gemini": _build_gemini,
    "google": _build_gemini,   # alias
    "groq": _build_groq,
}


def _resolve(cfg: dict) -> tuple[Any, str]:
    provider = cfg["provider"]
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Supported: {list(_BUILDERS)}")
    api_key_env = cfg.get("api_key_env", "GOOGLE_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"{api_key_env} not set for provider '{provider}'. Add it to your .env file."
        )
    return builder, api_key


def _build(cfg: dict, model: str, temperature: float, streaming: bool) -> BaseChatModel:
    builder, api_key = _resolve(cfg)
    llm = builder(
        model,
        temperature,
        cfg.get("max_tokens", 8192),
        streaming,
        api_key,
        float(cfg.get("timeout_seconds", 60)),
        int(cfg.get("retry_attempts", 2)),
    )
    llm.callbacks = [_token_tracker]
    return llm


def _with_fallback(cfg: dict, primary_model: str, temperature: float, streaming: bool) -> Runnable:
    """Primary model, replayed on the fallback model if a call fails."""
    primary = _build(cfg, primary_model, temperature, streaming)
    fallback_model = cfg.get("fallback_model")
    if not fallback_model or fallback_model == primary_model:
        return primary
    fallback = _build(cfg, fallback_model, temperature, streaming)
    logger.debug("LLM initialized: provider=%s model=%s fallback=%s",
                 cfg["provider"], primary_model, fallback_model)
    # RunnableWithFallbacks forwards with_structured_output() and bind_tools()
    # to both models, so callers use it exactly like a chat model.
    return primary.with_fallbacks([fallback])


def get_llm(temperature: float | None = None, streaming: bool = False) -> Runnable:
    """Return the chat model for the active provider, with call-level fallback."""
    cfg = get_llm_config()
    temp = temperature if temperature is not None else cfg.get("temperature", 0.1)
    return _with_fallback(cfg, cfg.get("model"), temp, streaming)


def get_judge_llm(temperature: float | None = None) -> Runnable:
    """Return the judge model for the LLM-as-judge evaluator.

    Uses llm.providers.<provider>.judge_model. If a provider has no judge
    model configured, the primary model judges its own pipeline, which is
    weaker: the judge shares the blind spots of the agents it grades.
    """
    cfg = get_llm_config()
    judge_model = cfg.get("judge_model")
    temp = temperature if temperature is not None else cfg.get("temperature", 0.1)
    if not judge_model:
        logger.warning(
            "No judge_model configured for provider '%s'; the primary model will "
            "grade its own decisions.", cfg["provider"],
        )
        return get_llm(temperature=temp)
    logger.debug("Judge LLM initialized: provider=%s model=%s", cfg["provider"], judge_model)
    return _with_fallback(cfg, judge_model, temp, False)


def get_structured_llm(schema: Any, temperature: float | None = None):
    """Return an LLM bound to a Pydantic schema for structured output."""
    return get_llm(temperature=temperature).with_structured_output(schema)

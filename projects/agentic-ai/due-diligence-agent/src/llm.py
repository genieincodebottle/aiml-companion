"""
LLM client factory with automatic fallback, retry, and token tracking.

Provides a single `get_llm()` entry point that returns a configured
ChatModel. Supports Google Gemini (free tier), OpenAI, and Ollama.

Usage:
    from src.llm import get_llm, get_structured_llm

    llm = get_llm()                                      # Default model
    structured = get_structured_llm(FinancialAnalysis)    # Guaranteed JSON output
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Type

from pydantic import BaseModel

from src.config import get_budget_config, get_model_config, get_pipeline_config

logger = logging.getLogger(__name__)


class TokenTracker:
    """Thread-safe token usage tracker."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_errors = 0
        self.call_log: list[dict] = []

    def record(self, input_tokens: int, output_tokens: int, model: str, duration: float):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1
        self.call_log.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": round(duration, 2),
        })

    def record_error(self):
        self.total_errors += 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def cost_estimate(self, input_rate: float | None = None, output_rate: float | None = None) -> float:
        """Estimate cost in USD based on per-1k-token rates (default: budget config)."""
        return estimate_cost(self.total_input_tokens, self.total_output_tokens, input_rate, output_rate)

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_estimate(), 6),
        }


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float | None = None,
    output_rate: float | None = None,
) -> float:
    """USD cost of a token count, using the per-1k rates in budget config."""
    budget = get_budget_config()
    if input_rate is None:
        input_rate = budget.get("cost_per_1k_input_tokens", 0.0001)
    if output_rate is None:
        output_rate = budget.get("cost_per_1k_output_tokens", 0.0004)
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


# Global tracker instance
token_tracker = TokenTracker()

# Per-node usage, so the guardrail budget can be charged for exactly the LLM
# calls one agent made even while four agents run in parallel. The global
# token_tracker cannot attribute usage to a node; this can.
_usage_scope: ContextVar[dict | None] = ContextVar("usage_scope", default=None)


class AgentLimitExceeded(RuntimeError):
    """An agent hit its per-agent token ceiling or was cancelled after a timeout.

    Raised before the next LLM call, so the agent's own try/except turns it into
    its normal degraded output instead of spending more.
    """


@contextmanager
def usage_scope(max_tokens: int | None = None) -> Iterator[dict]:
    """Collect the tokens and cost of every `invoke_with_tracking` call inside the block.

    Yields {"input_tokens", "output_tokens", "total_tokens", "cost_usd", ...}, updated
    live. `max_tokens` is the per-agent ceiling: once the scope has used that many
    tokens the next call raises AgentLimitExceeded. Setting usage["cancelled"] = True
    (done on a per-agent timeout) stops an abandoned agent at its next call.
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
             "limit": max_tokens, "cancelled": False}
    reset = _usage_scope.set(usage)
    try:
        yield usage
    finally:
        _usage_scope.reset(reset)


def _usage_of(message: Any) -> tuple[int, int]:
    """(input_tokens, output_tokens) reported by a chat response.

    `usage_metadata` is LangChain's provider-portable field and is where Gemini
    puts its counts. The old code read only `response_metadata`, which Gemini
    does not fill, and a structured-output chain returned a bare Pydantic object
    with no metadata at all, so every call recorded 0 tokens and the budget
    guardrail never moved. The legacy keys stay as a fallback.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    if usage.get("input_tokens") or usage.get("output_tokens"):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    meta = getattr(message, "response_metadata", None) or {}
    legacy = meta.get("usage_metadata") or meta.get("token_usage") or {}
    return (
        int(legacy.get("prompt_token_count", 0) or legacy.get("prompt_tokens", 0)),
        int(legacy.get("candidates_token_count", 0) or legacy.get("completion_tokens", 0)),
    )


def _get_google_llm(model_name: str, temperature: float, max_tokens: int):
    """Create a Google Gemini ChatModel."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def _get_openai_llm(model_name: str, temperature: float, max_tokens: int):
    """Create an OpenAI ChatModel."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _get_ollama_llm(model_name: str, temperature: float, max_tokens: int):
    """Create an Ollama (local) ChatModel."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
    )


def _get_offline_llm(model_name: str, temperature: float, max_tokens: int):
    """No key, no network. See src/llm_offline.py for what it does and does
    not do. It is a real part of the product, not a test double: it is how a
    reader sees the pipeline work before deciding to get a key."""
    from src.llm_offline import get_offline_llm

    return get_offline_llm(model_name, temperature, max_tokens)


_PROVIDER_FACTORY = {
    "google": _get_google_llm,
    "openai": _get_openai_llm,
    "ollama": _get_ollama_llm,
    "offline": _get_offline_llm,
}


def get_llm(
    provider: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Get a configured LLM instance with config defaults.

    Args:
        provider: Override provider (google | openai | ollama)
        model_name: Override model name
        temperature: Override temperature
        max_tokens: Override max output tokens

    Returns:
        A LangChain ChatModel instance.
    """
    cfg = get_model_config()

    # The environment wins, so `LLM_PROVIDER=offline` needs no config edit.
    provider = provider or os.getenv("LLM_PROVIDER") or cfg.get("provider", "google")
    model_name = model_name or cfg.get("name", "gemini-3.6-flash")
    temperature = temperature if temperature is not None else cfg.get("temperature", 0.1)
    max_tokens = max_tokens or cfg.get("max_output_tokens", 4096)

    factory = _PROVIDER_FACTORY.get(provider)
    if not factory:
        raise ValueError(f"Unknown LLM provider: {provider}. Use: {list(_PROVIDER_FACTORY.keys())}")

    try:
        llm = factory(model_name, temperature, max_tokens)
        logger.info("Created %s LLM: %s (temp=%.1f)", provider, model_name, temperature)
        return llm
    except Exception as e:
        # Try fallback model
        fallback = cfg.get("fallback_name")
        if fallback and fallback != model_name:
            logger.warning("Primary model %s failed (%s), trying fallback %s", model_name, e, fallback)
            return factory(fallback, temperature, max_tokens)
        raise


def get_structured_llm(
    schema: Type[BaseModel],
    temperature: float | None = None,
    **kwargs,
):
    """Get an LLM that returns structured Pydantic output.

    Args:
        schema: Pydantic model class for output structure.
        temperature: Override temperature.

    Returns:
        A chain that guarantees output matching the schema.
    """
    llm = get_llm(temperature=temperature, **kwargs)
    # include_raw=True keeps the raw message next to the parsed model; without it
    # the token usage is thrown away. invoke_with_tracking unwraps the result.
    return llm.with_structured_output(schema, include_raw=True)


def invoke_with_tracking(
    llm,
    prompt: str | list,
    agent_name: str = "unknown",
) -> Any:
    """Invoke LLM with automatic token tracking and retry.

    Args:
        llm: LangChain ChatModel (or structured output chain).
        prompt: String prompt or list of messages.
        agent_name: Name for logging.

    Returns:
        LLM response (string or Pydantic object).
    """
    cfg = get_model_config()
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_delay_seconds", 2)
    # pipeline.max_retries_per_agent counts retries, so attempts = 1 + retries.
    # It can only tighten model.max_retries (attempts), never raise it.
    per_agent_retries = get_pipeline_config().get("max_retries_per_agent")
    if per_agent_retries is not None:
        max_retries = max(1, min(max_retries, 1 + per_agent_retries))

    scope = _usage_scope.get()
    if scope is not None:
        if scope["cancelled"]:
            raise AgentLimitExceeded(f"[{agent_name}] cancelled after the per-agent timeout")
        if scope["limit"] and scope["total_tokens"] >= scope["limit"]:
            raise AgentLimitExceeded(
                f"[{agent_name}] per-agent token ceiling reached "
                f"({scope['total_tokens']}/{scope['limit']})"
            )

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = llm.invoke(prompt)
            duration = time.time() - start

            # A structured chain built with include_raw=True returns
            # {"raw", "parsed", "parsing_error"}; usage lives on the raw message.
            structured = isinstance(response, dict) and "raw" in response
            input_tokens, output_tokens = _usage_of(response["raw"] if structured else response)

            token_tracker.record(input_tokens, output_tokens, agent_name, duration)
            scope = _usage_scope.get()
            if scope is not None:
                scope["input_tokens"] += input_tokens
                scope["output_tokens"] += output_tokens
                scope["total_tokens"] += input_tokens + output_tokens
                scope["cost_usd"] += estimate_cost(input_tokens, output_tokens)
            logger.debug(
                "[%s] LLM call: %d in + %d out tokens, %.1fs",
                agent_name, input_tokens, output_tokens, duration,
            )
            if structured:
                if response.get("parsing_error") and response.get("parsed") is None:
                    raise ValueError(f"Structured output failed to parse: {response['parsing_error']}")
                return response["parsed"]
            return response

        except Exception as e:
            token_tracker.record_error()
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "[%s] LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    agent_name, attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("[%s] LLM call failed after %d attempts: %s", agent_name, max_retries, e)
                raise

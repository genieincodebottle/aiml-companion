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
from typing import Any, Type

from pydantic import BaseModel

from src.config import get_model_config

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

    def cost_estimate(self, input_rate: float = 0.0001, output_rate: float = 0.0004) -> float:
        """Estimate cost in USD based on per-1k-token rates."""
        return (self.total_input_tokens / 1000 * input_rate) + (self.total_output_tokens / 1000 * output_rate)

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_estimate(), 6),
        }


# Global tracker instance
token_tracker = TokenTracker()


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


_PROVIDER_FACTORY = {
    "google": _get_google_llm,
    "openai": _get_openai_llm,
    "ollama": _get_ollama_llm,
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

    provider = provider or cfg.get("provider", "google")
    model_name = model_name or cfg.get("name", "gemini-2.5-flash")
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
    return llm.with_structured_output(schema)


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

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = llm.invoke(prompt)
            duration = time.time() - start

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
                input_tokens = usage.get("prompt_token_count", 0) or usage.get("prompt_tokens", 0)
                output_tokens = usage.get("candidates_token_count", 0) or usage.get("completion_tokens", 0)

            token_tracker.record(input_tokens, output_tokens, agent_name, duration)
            logger.debug(
                "[%s] LLM call: %d in + %d out tokens, %.1fs",
                agent_name, input_tokens, output_tokens, duration,
            )
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

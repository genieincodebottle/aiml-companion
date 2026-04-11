"""
LLM factory - returns a configured ChatGoogleGenerativeAI instance.
Three-layer fallback: primary model -> fallback model -> raise.
"""

from __future__ import annotations

import os
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_llm_config

logger = logging.getLogger(__name__)

_PRIMARY_MODEL = "gemini-2.0-flash"
_FALLBACK_MODEL = "gemini-1.5-flash"


def get_llm(temperature: float | None = None, streaming: bool = False) -> ChatGoogleGenerativeAI:
    """
    Returns a configured Gemini LLM.
    Uses primary model with automatic fallback on failure.
    """
    cfg = get_llm_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key."
        )

    kwargs = dict(
        model=cfg.get("model", _PRIMARY_MODEL),
        google_api_key=api_key,
        temperature=temperature if temperature is not None else cfg.get("temperature", 0.1),
        max_output_tokens=cfg.get("max_tokens", 8192),
        streaming=streaming,
    )

    try:
        llm = ChatGoogleGenerativeAI(**kwargs)
        logger.debug(f"LLM initialized: {kwargs['model']}")
        return llm
    except Exception as e:
        logger.warning(f"Primary model {kwargs['model']} failed ({e}), using fallback {_FALLBACK_MODEL}")
        kwargs["model"] = _FALLBACK_MODEL
        return ChatGoogleGenerativeAI(**kwargs)


def get_structured_llm(schema, temperature: float | None = None):
    """Returns an LLM bound to a Pydantic schema for structured output."""
    return get_llm(temperature=temperature).with_structured_output(schema)

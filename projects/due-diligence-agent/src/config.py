"""
Configuration loader with caching, environment variable overrides, and safe defaults.

Usage:
    from src.config import load_config, get_agent_config, get_pipeline_config

    config = load_config()                          # Load from configs/base.yaml
    config = load_config("configs/production.yaml") # Load specific file
    agent_cfg = get_agent_config("financial_analyst")
    pipeline_cfg = get_pipeline_config()
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

_config: dict | None = None


def _default_config() -> dict:
    """Hardcoded fallback - pipeline never crashes due to missing config file."""
    return {
        "model": {
            "provider": "google",
            "name": "gemini-2.5-flash",
            "fallback_name": "gemini-2.0-flash",
            "temperature": 0.1,
            "max_output_tokens": 4096,
            "timeout_seconds": 60,
            "max_retries": 3,
            "retry_delay_seconds": 2,
        },
        "agents": {
            "lead_analyst": {"role": "Lead Due Diligence Analyst", "temperature": 0.2, "max_tokens": 4096},
            "financial_analyst": {"role": "Financial Analyst", "temperature": 0.1, "max_tokens": 3072},
            "news_sentiment": {"role": "News & Sentiment Analyst", "temperature": 0.2, "max_tokens": 3072},
            "competitive_intel": {"role": "Competitive Intelligence Analyst", "temperature": 0.2, "max_tokens": 3072},
            "risk_assessor": {"role": "Risk Assessment Specialist", "temperature": 0.1, "max_tokens": 3072},
            "fact_checker": {"role": "Fact Verification Specialist", "temperature": 0.0, "max_tokens": 2048},
        },
        "pipeline": {
            "max_parallel_agents": 4,
            "max_retries_per_agent": 2,
            "max_debate_rounds": 2,
            "max_fact_check_depth": 2,
            "timeout_per_agent_seconds": 120,
            "enable_self_correction": True,
            "enable_debate": True,
        },
        "budget": {
            "max_total_tokens": 100000,
            "max_tokens_per_agent": 20000,
            "warning_threshold_pct": 80,
            "cost_per_1k_input_tokens": 0.0001,
            "cost_per_1k_output_tokens": 0.0004,
            "max_cost_usd": 0.50,
        },
        "search": {
            "provider": "tavily",
            "max_results_per_query": 5,
            "max_queries_per_agent": 3,
            "search_depth": "advanced",
        },
        "cache": {"enabled": True, "backend": "sqlite", "ttl_hours": 24, "max_entries": 1000},
        "guardrails": {
            "max_reasoning_iterations": 10,
            "max_agent_calls": 30,
            "max_execution_time_seconds": 300,
            "hallucination_check_enabled": True,
            "confidence_threshold": 0.6,
            "pii_detection_enabled": True,
            "source_grounding_required": True,
        },
        "output": {
            "format": "markdown",
            "include_sources": True,
            "include_confidence_scores": True,
            "include_agent_traces": True,
            "include_cost_breakdown": True,
            "reports_dir": "artifacts/reports",
        },
        "logging": {"level": "INFO", "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    }


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration from YAML with environment variable overrides.

    Args:
        path: Path to YAML config file. None uses default.

    Returns:
        Configuration dictionary.
    """
    global _config
    if _config is not None and path is None:
        return _config

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)
    except FileNotFoundError:
        logger.warning("Config file not found at %s, using defaults", config_path)
        config = _default_config()
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in %s: %s, using defaults", config_path, e)
        config = _default_config()

    # Environment variable overrides (flat mapping)
    env_overrides = {
        "DD_MODEL_PROVIDER": ("model", "provider"),
        "DD_MODEL_NAME": ("model", "name"),
        "DD_MODEL_TEMPERATURE": ("model", "temperature", float),
        "DD_MAX_COST_USD": ("budget", "max_cost_usd", float),
        "DD_MAX_TOTAL_TOKENS": ("budget", "max_total_tokens", int),
        "DD_LOG_LEVEL": ("logging", "level"),
        "DD_SEARCH_PROVIDER": ("search", "provider"),
        "DD_CACHE_ENABLED": ("cache", "enabled", lambda v: v.lower() == "true"),
    }

    for env_key, mapping in env_overrides.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            section, key = mapping[0], mapping[1]
            cast_fn = mapping[2] if len(mapping) > 2 else str
            if section not in config:
                config[section] = {}
            config[section][key] = cast_fn(env_val)
            logger.debug("Override %s.%s = %s from env %s", section, key, env_val, env_key)

    if path is None:
        _config = config

    return config


def reset_config() -> None:
    """Clear cached config (useful for testing)."""
    global _config
    _config = None


def get_agent_config(agent_name: str) -> dict:
    """Get configuration for a specific agent."""
    config = load_config()
    agents = config.get("agents", {})
    if agent_name not in agents:
        logger.warning("No config for agent '%s', using defaults", agent_name)
        return {"role": agent_name, "temperature": 0.1, "max_tokens": 3072}
    return agents[agent_name]


def get_pipeline_config() -> dict:
    """Get pipeline execution settings."""
    config = load_config()
    return config.get("pipeline", _default_config()["pipeline"])


def get_budget_config() -> dict:
    """Get token/cost budget settings."""
    config = load_config()
    return config.get("budget", _default_config()["budget"])


def get_search_config() -> dict:
    """Get search tool settings."""
    config = load_config()
    return config.get("search", _default_config()["search"])


def get_guardrails_config() -> dict:
    """Get guardrails settings."""
    config = load_config()
    return config.get("guardrails", _default_config()["guardrails"])


def get_output_config() -> dict:
    """Get output/report settings."""
    config = load_config()
    return config.get("output", _default_config()["output"])


def get_model_config() -> dict:
    """Get LLM model settings."""
    config = load_config()
    return config.get("model", _default_config()["model"])

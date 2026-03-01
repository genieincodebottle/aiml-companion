# ============================================
# YAML Configuration Loader
# ============================================
# Loads agent configs from configs/base.yaml.
# Teaches config-driven architecture: add/remove agents without code changes.

import os
import yaml
import logging

logger = logging.getLogger(__name__)

_config = None
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


def load_config(path: str = None) -> dict:
    """Load and cache the YAML configuration."""
    global _config
    if _config is not None and path is None:
        return _config
    config_path = path or _CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return _config
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using defaults")
        return _default_config()


def _default_config() -> dict:
    """Fallback config when YAML is missing."""
    return {
        "model": {"name": "gemini-2.5-flash", "provider": "google"},
        "budget": {"token_budget": 50000, "warn_at_percent": 80},
        "search": {"max_results": 5, "timeout_seconds": 10},
        "pipeline": {
            "max_sub_topics": 3,
            "max_revisions": 2,
            "quality_threshold": 0.4,
            "review_pass_score": 7,
        },
    }


def get_model_name() -> str:
    """Get the configured LLM model name."""
    cfg = load_config()
    return cfg.get("model", {}).get("name", "gemini-2.5-flash")


def get_agent_config(agent_name: str) -> dict:
    """Get config for a specific agent (temperature, max_tokens, etc.)."""
    cfg = load_config()
    return cfg.get("agents", {}).get(agent_name, {})


def get_pipeline_config() -> dict:
    """Get pipeline-level config (max sub-topics, revisions, thresholds)."""
    cfg = load_config()
    return cfg.get("pipeline", {
        "max_sub_topics": 3,
        "max_revisions": 2,
        "quality_threshold": 0.4,
        "review_pass_score": 7,
    })


def get_budget_config() -> dict:
    """Get token budget configuration."""
    cfg = load_config()
    return cfg.get("budget", {"token_budget": 50000, "warn_at_percent": 80})

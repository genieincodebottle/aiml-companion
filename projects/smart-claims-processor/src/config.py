"""
Configuration loader - YAML base + environment variable overrides.
Access via get_config(), get_agent_config(), get_hitl_config(), etc.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "base.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    return _load_raw()


def get_llm_config() -> dict:
    cfg = _load_raw()["llm"]
    # Env overrides
    if os.getenv("LLM_MODEL"):
        cfg["model"] = os.getenv("LLM_MODEL")
    if os.getenv("LLM_TEMPERATURE"):
        cfg["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
    return cfg


def get_agent_config(agent_name: str) -> dict:
    return _load_raw()["agents"].get(agent_name, {})


def get_hitl_config() -> dict:
    cfg = _load_raw()["hitl"]
    # Env overrides
    if os.getenv("HITL_MIN_AMOUNT"):
        cfg["triggers"]["min_amount_usd"] = float(os.getenv("HITL_MIN_AMOUNT"))
    if os.getenv("HITL_FRAUD_THRESHOLD"):
        cfg["triggers"]["fraud_score"] = float(os.getenv("HITL_FRAUD_THRESHOLD"))
    if os.getenv("HITL_LOW_CONFIDENCE"):
        cfg["triggers"]["low_confidence"] = float(os.getenv("HITL_LOW_CONFIDENCE"))
    return cfg


def get_guardrails_config() -> dict:
    cfg = _load_raw()["guardrails"]
    if os.getenv("MAX_TOKENS_PER_CLAIM"):
        cfg["max_tokens_per_claim"] = int(os.getenv("MAX_TOKENS_PER_CLAIM"))
    if os.getenv("MAX_COST_PER_CLAIM"):
        cfg["max_cost_usd"] = float(os.getenv("MAX_COST_PER_CLAIM"))
    if os.getenv("MAX_AGENT_CALLS"):
        cfg["max_agent_calls"] = int(os.getenv("MAX_AGENT_CALLS"))
    return cfg


def get_security_config() -> dict:
    cfg = _load_raw()["security"]
    if os.getenv("AUDIT_LOG_PATH"):
        cfg["audit_log"]["path"] = os.getenv("AUDIT_LOG_PATH")
    cfg["pii_masking"] = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
    return cfg


def get_evaluation_config() -> dict:
    return _load_raw()["evaluation"]


def get_pipeline_config() -> dict:
    return _load_raw()["pipeline"]


def get_output_config() -> dict:
    return _load_raw()["output"]

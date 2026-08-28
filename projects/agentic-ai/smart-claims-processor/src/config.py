"""
Configuration loader - .env for provider/country, YAML for everything else.

Hierarchy (highest precedence first):
  1. Runtime override (via /api/settings endpoints)
  2. .env (LLM_PROVIDER, COUNTRY, API keys, threshold overrides)
  3. Country profile YAML (configs/countries/{country}.yaml)
  4. Base YAML (configs/base.yaml) - model IDs, tunables, agent config

Access via get_config(), get_country_config(), get_llm_config(), etc.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "configs"
CONFIG_PATH = _CONFIG_DIR / "base.yaml"
_COUNTRIES_DIR = _CONFIG_DIR / "countries"


# ── YAML loaders ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def _load_country_yaml(code: str) -> dict:
    path = _COUNTRIES_DIR / f"{code.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Country profile not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay wins on leaf conflicts."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── Runtime overrides (mutable, process lifetime) ───────────────────────────

_runtime_overrides: dict = {}


def set_llm_provider_override(provider: str | None) -> None:
    if provider is None:
        _runtime_overrides.pop("llm_provider", None)
    else:
        _runtime_overrides["llm_provider"] = provider


def set_country_override(code: str | None) -> None:
    if code is None:
        _runtime_overrides.pop("country", None)
    else:
        _runtime_overrides["country"] = code.lower()


# ── Country resolution ──────────────────────────────────────────────────────

def _active_country_code() -> str:
    """Resolve active country: runtime > env > fallback 'india'."""
    return (
        _runtime_overrides.get("country")
        or os.getenv("COUNTRY", "india").strip().lower()
    )


def get_available_countries() -> list[str]:
    """Return list of country codes that have a YAML profile."""
    return sorted(p.stem for p in _COUNTRIES_DIR.glob("*.yaml"))


# ── Public config accessors ─────────────────────────────────────────────────

def get_config() -> dict:
    return _load_raw()


def get_country_config() -> dict:
    """Full country profile for the active country."""
    code = _active_country_code()
    return _load_country_yaml(code)


def get_country_meta() -> dict:
    """Just the country.* block (code, name, currency, regulator, etc.)."""
    return get_country_config().get("country", {})


def get_llm_config() -> dict:
    cfg = dict(_load_raw()["llm"])
    provider = (
        _runtime_overrides.get("llm_provider")
        or os.getenv("LLM_PROVIDER", "groq")
    )
    cfg["provider"] = provider.lower()
    providers = cfg.get("providers", {})
    provider_cfg = providers.get(cfg["provider"], {})
    cfg["model"] = os.getenv("LLM_MODEL") or provider_cfg.get("model")
    cfg["fallback_model"] = provider_cfg.get("fallback_model")
    cfg["api_key_env"] = provider_cfg.get("api_key_env", "GOOGLE_API_KEY")
    if os.getenv("LLM_TEMPERATURE"):
        cfg["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
    return cfg


def get_agent_config(agent_name: str) -> dict:
    return _load_raw()["agents"].get(agent_name, {})


def get_hitl_config() -> dict:
    base = copy.deepcopy(_load_raw()["hitl"])
    # Merge country-level HITL overrides
    country_hitl = get_country_config().get("hitl", {})
    cfg = _deep_merge(base, country_hitl)
    # Env overrides
    if os.getenv("HITL_MIN_AMOUNT"):
        cfg["triggers"]["min_amount"] = float(os.getenv("HITL_MIN_AMOUNT"))
    if os.getenv("HITL_FRAUD_THRESHOLD"):
        cfg["triggers"]["fraud_score"] = float(os.getenv("HITL_FRAUD_THRESHOLD"))
    if os.getenv("HITL_LOW_CONFIDENCE"):
        cfg["triggers"]["low_confidence"] = float(os.getenv("HITL_LOW_CONFIDENCE"))
    return cfg


def get_pii_config() -> dict:
    """PII masking config from the active country profile."""
    return get_country_config().get("pii", {})


def get_depreciation_config() -> dict:
    """Depreciation config from the active country profile."""
    return get_country_config().get("depreciation", {})


def get_settlement_config() -> dict:
    """Settlement rules from the active country profile."""
    return get_country_config().get("settlement", {})


def get_communication_config() -> dict:
    """Communication templates + contact info from the active country profile."""
    return get_country_config().get("communication", {})


def get_fraud_baselines() -> dict:
    """Fraud statistical baselines from the active country profile."""
    return get_country_config().get("fraud_baselines", {})


def get_coverage_mapping() -> dict:
    """Claim type -> coverage category mapping from the active country."""
    return get_country_config().get("coverage_mapping", {})


def get_required_documents(claim_type: str) -> list[str]:
    """Required documents for a claim type in the active country."""
    docs = get_country_config().get("required_documents", {})
    return docs.get(claim_type, [])


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


def get_confidence_gate_config() -> dict:
    """Confidence gate thresholds for per-agent HITL routing."""
    return _load_raw().get("confidence_gates", {"enabled": False})


def get_pipeline_config() -> dict:
    return _load_raw()["pipeline"]


def get_output_config() -> dict:
    return _load_raw()["output"]

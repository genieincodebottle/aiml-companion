"""
Runtime settings - LLM provider + country profile.

GET  /api/settings/llm       - Current + available LLM providers.
PUT  /api/settings/llm       - Admin-only; switch the active provider.
GET  /api/settings/country   - Current + available country profiles.
PUT  /api/settings/country   - Admin-only; switch the active country.
"""
from __future__ import annotations # for Python 3.10-3.11 compatibility, allows forward references in type hints without quotes

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.security import get_current_user, require_role
from src.config import (
    get_available_countries,
    get_country_config,
    get_country_meta,
    get_llm_config,
    set_country_override,
    set_llm_provider_override,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])

SUPPORTED_PROVIDERS = ("gemini", "groq")


# --- LLM Provider ---------------------------------------------

class LLMProviderUpdate(BaseModel):
    provider: str


def _describe_provider(provider: str) -> dict:
    """Helper to return details about an LLM provider, including which environment variable is used for the API key."""
    
    from src.config import get_config
    cfg = get_config()["llm"]
    info = cfg.get("providers", {}).get(provider, {})
    key_env = info.get("api_key_env", "")
    return {
        "provider": provider,
        "model": info.get("model"),
        "fallback_model": info.get("fallback_model"),
        "api_key_env": key_env,
        "api_key_set": bool(os.getenv(key_env)) if key_env else False,
    }


@router.get("/llm")
def get_llm_settings(_=Depends(get_current_user)):
    """Return active LLM provider + list of available providers. The active provider controls 
    which underlying LLM is used for agent reasoning, evaluations, and other tasks."""
    
    active = get_llm_config()
    return {
        "active": {
            "provider": active.get("provider"),
            "model": active.get("model"),
            "fallback_model": active.get("fallback_model"),
            "temperature": active.get("temperature"),
        },
        "available": [_describe_provider(p) for p in SUPPORTED_PROVIDERS],
    }


@router.put("/llm")
def update_llm_settings(
    body: LLMProviderUpdate,
    _=Depends(require_role("admin")),
):
    """Set the active LLM provider. This controls which underlying LLM is used for 
    agent reasoning, evaluations, and other tasks.
    Note: changing the provider may require setting the appropriate 
    API key in the environment variable specified by the provider's configuration (e.g. API_KEY_ENV)."""

    provider = body.provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}",
        )
    set_llm_provider_override(provider)
    active = get_llm_config()
    return {
        "ok": True,
        "active": {
            "provider": active.get("provider"),
            "model": active.get("model"),
            "fallback_model": active.get("fallback_model"),
        },
    }


# -- Country Profile ---------------------------------------------

class CountryUpdate(BaseModel):
    country: str


@router.get("/country")
def get_country_settings(_=Depends(get_current_user)):
    """Return active country profile + list of available countries."""

    meta = get_country_meta()
    cfg = get_country_config()
    available = get_available_countries()
    return {
        "active": {
            "code": meta.get("code"),
            "name": meta.get("name"),
            "currency": meta.get("currency"),
            "currency_symbol": meta.get("currency_symbol"),
            "regulator": meta.get("regulator"),
        },
        "claim_types": cfg.get("claim_types", []),
        "depreciation_method": cfg.get("depreciation", {}).get("method"),
        "available": available,
    }


@router.put("/country")
def update_country_settings(
    body: CountryUpdate,
    _=Depends(require_role("admin")),
):
    """Set the active country profile. This controls claim type definitions, depreciation rules, 
    and other country-specific logic."""

    code = body.country.lower().strip()
    available = get_available_countries()
    if code not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown country '{code}'. Available: {available}",
        )
    set_country_override(code)
    meta = get_country_meta()
    return {
        "ok": True,
        "active": {
            "code": meta.get("code"),
            "name": meta.get("name"),
            "currency": meta.get("currency"),
            "currency_symbol": meta.get("currency_symbol"),
        },
    }

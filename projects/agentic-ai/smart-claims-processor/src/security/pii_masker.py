"""
PII Masking Layer - country-aware, config-driven.

Loads PII field lists + regex patterns from the active country profile
(configs/countries/{code}.yaml -> pii block). Falls back to a safe default
if no country config is loaded yet.

Replaces real PII with deterministic placeholders so:
  1. The agent can still reason about the claim (e.g. "CLAIMANT_NAME filed on INCIDENT_DATE")
  2. Real data never leaves our infrastructure in LLM prompts
  3. De-masking is possible from the original claim object (held in secure memory)

US: masks SSN, driver license, US phone, ZIP
India: masks Aadhaar, PAN, Indian mobile, pincode
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults (used if country config isn't available yet) ────────────────────

_DEFAULT_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PHONE": re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
    "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "DOB": re.compile(r"\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b"),
}

_NAME_FIELDS = {"claimant_name", "name", "insured_name", "beneficiary_name"}
_DEFAULT_REDACT_FIELDS = {"claimant_dob", "ssn", "bank_account", "credit_card", "password"}


def _get_country_pii_config() -> dict:
    """Load PII config from active country. Return empty dict on failure."""
    try:
        from src.config import get_pii_config
        return get_pii_config()
    except Exception:
        return {}


def _build_patterns() -> dict[str, re.Pattern]:
    """Build regex patterns from country config, falling back to defaults."""
    pii = _get_country_pii_config()
    custom = pii.get("patterns", {})
    if not custom:
        return dict(_DEFAULT_PATTERNS)

    patterns: dict[str, re.Pattern] = {}
    for label, regex_str in custom.items():
        try:
            patterns[label.upper()] = re.compile(regex_str)
        except re.error as e:
            logger.warning("Invalid PII regex for %s: %s", label, e)
    return patterns


def _get_redact_fields() -> set[str]:
    """Redact fields from country config, union with hardcoded safety set."""
    pii = _get_country_pii_config()
    country_redact = set(pii.get("redact_fields", []))
    return _DEFAULT_REDACT_FIELDS | country_redact


# ── Public API ───────────────────────────────────────────────────────────────

def mask_text(text: str) -> str:
    """Apply regex-based PII masking to a string."""
    if not text or not isinstance(text, str):
        return text
    patterns = _build_patterns()
    result = text
    for label, pattern in patterns.items():
        result = pattern.sub(f"[{label}]", result)
    return result


def mask_claim(claim: dict) -> dict:
    """Deep-copy a claim dict and mask all PII. Safe for LLM consumption."""
    masked = copy.deepcopy(claim)
    redact_fields = _get_redact_fields()
    patterns = _build_patterns()
    _mask_dict_recursive(masked, redact_fields, patterns, path="")
    return masked


def _mask_dict_recursive(
    obj: Any,
    redact_fields: set[str],
    patterns: dict[str, re.Pattern],
    path: str,
) -> None:
    """In-place recursive masking."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{path}.{key}" if path else key
            field_key = key.lower()

            if field_key in redact_fields:
                obj[key] = "[REDACTED]"
            elif field_key in _NAME_FIELDS:
                if isinstance(value, str) and value:
                    obj[key] = "[CLAIMANT_NAME]"
            elif isinstance(value, str):
                result = value
                for label, pattern in patterns.items():
                    result = pattern.sub(f"[{label}]", result)
                obj[key] = result
            else:
                _mask_dict_recursive(value, redact_fields, patterns, field_path)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                result = item
                for label, pattern in patterns.items():
                    result = pattern.sub(f"[{label}]", result)
                obj[i] = result
            else:
                _mask_dict_recursive(item, redact_fields, patterns, path)


def get_masked_summary(claim: dict) -> str:
    """Natural-language summary with PII masked. Safe for LLM prompts."""
    from src.utils import currency_symbol
    symbol = currency_symbol()

    masked = mask_claim(claim)
    return (
        f"Claim ID: {masked.get('claim_id', 'UNKNOWN')} | "
        f"Policy: {masked.get('policy_number', 'UNKNOWN')} | "
        f"Claimant: {masked.get('claimant_name', '[CLAIMANT_NAME]')} | "
        f"Incident: {masked.get('incident_type', 'UNKNOWN')} on {masked.get('incident_date', 'UNKNOWN')} | "
        f"Location: {masked.get('incident_location', 'UNKNOWN')} | "
        f"Amount: {symbol}{masked.get('estimated_amount', 0):,.2f} | "
        f"Description: {masked.get('incident_description', 'N/A')}"
    )

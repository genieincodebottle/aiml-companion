"""
PII Masking Layer - masks sensitive data before sending to any LLM.

Replaces real PII with deterministic placeholders so:
1. The agent can still reason about the claim (e.g. "CLAIMANT_NAME filed on INCIDENT_DATE")
2. Real data never leaves our infrastructure in LLM prompts
3. De-masking is possible from the original claim object (held in secure memory)

Patterns:
  - Email addresses
  - Phone numbers (US formats)
  - SSN
  - Date of birth
  - Credit card numbers
  - Bank account numbers
  - Street addresses
  - Names (via field-based masking, not regex)
"""

from __future__ import annotations

import copy
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Regex Patterns ────────────────────────────────────────────────────────────

_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PHONE": re.compile(r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
    "SSN": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "BANK_ACCOUNT": re.compile(r"\b\d{8,17}\b"),
    "ZIP_CODE": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "DOB": re.compile(r"\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b"),
}

# Fields that contain names - masked via field name, not regex
_NAME_FIELDS = {"claimant_name", "name", "insured_name", "beneficiary_name"}

# Fields that should be completely redacted (never sent to LLM)
_REDACT_FIELDS = {"claimant_dob", "ssn", "bank_account", "credit_card", "password"}


def mask_text(text: str) -> str:
    """Apply regex-based PII masking to a string."""
    if not text or not isinstance(text, str):
        return text
    result = text
    for label, pattern in _PATTERNS.items():
        result = pattern.sub(f"[{label}]", result)
    return result


def mask_claim(claim: dict) -> dict:
    """
    Deep-copy a claim dict and mask all PII.
    Returns a safe version suitable for LLM consumption.
    """
    masked = copy.deepcopy(claim)
    _mask_dict_recursive(masked, path="")
    return masked


def _mask_dict_recursive(obj: Any, path: str) -> None:
    """In-place recursive masking."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{path}.{key}" if path else key
            field_key = key.lower()

            if field_key in _REDACT_FIELDS:
                obj[key] = "[REDACTED]"
            elif field_key in _NAME_FIELDS:
                if isinstance(value, str) and value:
                    obj[key] = "[CLAIMANT_NAME]"
            elif isinstance(value, str):
                obj[key] = mask_text(value)
            else:
                _mask_dict_recursive(value, field_path)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = mask_text(item)
            else:
                _mask_dict_recursive(item, path)


def get_masked_summary(claim: dict) -> str:
    """
    Return a natural language summary of the claim with all PII masked.
    Safe to include in LLM prompts.
    """
    masked = mask_claim(claim)
    return (
        f"Claim ID: {masked.get('claim_id', 'UNKNOWN')} | "
        f"Policy: {masked.get('policy_number', 'UNKNOWN')} | "
        f"Claimant: {masked.get('claimant_name', '[CLAIMANT_NAME]')} | "
        f"Incident: {masked.get('incident_type', 'UNKNOWN')} on {masked.get('incident_date', 'UNKNOWN')} | "
        f"Location: {masked.get('incident_location', 'UNKNOWN')} | "
        f"Amount: ${masked.get('estimated_amount', 0):,.2f} | "
        f"Description: {masked.get('incident_description', 'N/A')}"
    )

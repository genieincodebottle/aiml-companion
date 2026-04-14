"""
Shared utilities used across agents and tools.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def currency_symbol() -> str:
    """Return the currency symbol for the active country config."""
    try:
        from src.config import get_country_meta
        return get_country_meta().get("currency_symbol", "$")
    except Exception:
        return "$"


_AUTO_KEYWORDS = {
    "auto", "vehicle", "car", "collision", "own_damage", "own damage",
    "third_party", "third party", "theft", "comprehensive", "motor",
}


def detect_asset_type(incident_type: str) -> str:
    """Return 'auto' or 'property' based on the incident type string."""
    it = incident_type.lower().replace("_", " ")
    return "auto" if any(k in it for k in _AUTO_KEYWORDS) else "property"


def calculate_asset_age(vehicle_year: int | str | None) -> int:
    """Return asset age in years from the vehicle year. Returns 0 if unknown."""
    if not vehicle_year:
        return 0
    from datetime import date
    return date.today().year - int(vehicle_year)


def recall_similar_claims(description: str, k: int = 3) -> str:
    """Retrieve similar past claims from memory. Returns formatted context string or empty string."""
    try:
        from src.memory.manager import memory
        similar = memory.recall_similar_claims(description, k=k)
        if not similar:
            return ""
        lines = ["SIMILAR PAST CLAIMS (from long-term memory):"]
        for s in similar:
            m = s.get("metadata", {})
            lines.append(
                f"  - {s['claim_id']}: decision={m.get('decision', '?')}, "
                f"settlement={m.get('settlement_amount', '?')}, "
                f"fraud_score={m.get('fraud_score', '?')}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Memory recall skipped: %s", e)
        return ""

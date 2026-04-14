"""
Fraud Pattern Database Tools - used by the CrewAI Fraud Detection Crew.

Contains:
- Known fraud patterns (rule-based first pass before LLM), country-aware
- Statistical baseline data for anomaly detection (US in USD, India in INR)
- Helper functions for each crew member

Patterns are split into:
  COMMON_PATTERNS  - apply to all countries
  US_PATTERNS      - US-specific (auto_collision, property_fire, etc.)
  INDIA_PATTERNS   - India-specific (own_damage, third_party, theft, etc.)

At runtime, `get_patterns()` returns common + country-specific patterns.
"""

from __future__ import annotations


# ── Helper Functions ──────────────────────────────────────────────────────────

def _active_country() -> str:
    try:
        from src.config import get_country_meta
        return get_country_meta().get("code", "us")
    except Exception:
        return "us"


def _is_india() -> bool:
    return _active_country() == "india"


def _days_after_start(claim: dict, policy: dict) -> int:
    from datetime import date
    try:
        start = date.fromisoformat(policy.get("start_date", "2000-01-01"))
        incident = date.fromisoformat(claim.get("incident_date", "2000-01-01"))
        return max(0, (incident - start).days)
    except ValueError:
        return 999


def _days_since_incident(claim: dict) -> int:
    """Days between incident and when the claim was filed (approximated by today)."""
    from datetime import date
    try:
        incident = date.fromisoformat(claim.get("incident_date", "2000-01-01"))
        return max(0, (date.today() - incident).days)
    except ValueError:
        return 0


def _is_weekend(date_str: str) -> bool:
    from datetime import date
    try:
        d = date.fromisoformat(date_str)
        return d.weekday() >= 5  # 5=Saturday, 6=Sunday
    except ValueError:
        return False


def _amount_exceeds_vehicle_value(claim: dict) -> bool:
    """Country-aware vehicle value check."""
    from datetime import date
    year = claim.get("vehicle_year")
    amount = float(claim.get("estimated_amount", 0))
    if not year:
        return False
    age = date.today().year - int(year)
    if _is_india():
        # India: new car ~₹8,00,000, depreciates ~15%/yr
        est_value = max(50_000, 800_000 * (0.85 ** age))
    else:
        # US: new car ~$35,000, depreciates ~15%/yr
        est_value = max(3_000, 35_000 * (0.85 ** age))
    return amount > est_value * 1.2  # Flag if >120% of estimated value


def _is_auto_type(claim: dict) -> bool:
    """Check if incident type is any auto/vehicle type (US or India)."""
    it = claim.get("incident_type", "").lower()
    auto_keywords = {
        "auto", "collision", "vehicle", "car", "own_damage", "own damage",
        "third_party", "third party", "theft", "comprehensive", "motor",
    }
    return any(k in it for k in auto_keywords)


def _description_word_count(claim: dict) -> int:
    desc = claim.get("incident_description", "")
    return len(desc.split())


def _document_count(claim: dict) -> int:
    docs = claim.get("documents", [])
    return len(docs) if isinstance(docs, list) else 0


def _is_high_value(claim: dict) -> bool:
    """Country-aware high-value check."""
    amount = float(claim.get("estimated_amount", 0))
    if _is_india():
        return amount >= 500_000  # ₹5 lakh
    return amount >= 10_000  # $10K


# ── Common Patterns (all countries) ──────────────────────────────────────────

COMMON_PATTERNS = [
    {
        "id": "FP-001",
        "name": "New Policy Quick Claim",
        "description": "Claim filed within 30 days of policy start - common fraud indicator",
        "risk_weight": 0.6,
        "check": lambda claim, policy: _days_after_start(claim, policy) <= 30,
    },
    {
        "id": "FP-002",
        "name": "Round Number Amount",
        "description": "Estimated amount is exactly a round number (staged claim indicator)",
        "risk_weight": 0.3,
        "check": lambda claim, policy: float(claim.get("estimated_amount", 0)) % 1000 == 0,
    },
    {
        "id": "FP-003",
        "name": "Weekend/Holiday Incident",
        "description": "Incident occurred on a weekend - higher proportion of fraudulent claims",
        "risk_weight": 0.2,
        "check": lambda claim, policy: _is_weekend(claim.get("incident_date", "")),
    },
    {
        "id": "FP-004",
        "name": "Repeat High-Value Claims",
        "description": "Multiple large claims from same policy holder",
        "risk_weight": 0.7,
        "check": lambda claim, policy: policy.get("claims_count", 0) >= 2,
    },
    {
        "id": "FP-005",
        "name": "Amount Exceeds Vehicle Value",
        "description": "Claimed damage exceeds estimated market value of vehicle",
        "risk_weight": 0.65,
        "check": lambda claim, policy: _amount_exceeds_vehicle_value(claim),
    },
    {
        "id": "FP-006",
        "name": "Late Reporting",
        "description": "Claim filed more than 15 days after incident - may indicate fabrication",
        "risk_weight": 0.45,
        "check": lambda claim, policy: _days_since_incident(claim) > 15,
    },
    {
        "id": "FP-007",
        "name": "Vague Incident Description",
        "description": "Incident description is suspiciously brief (fewer than 10 words)",
        "risk_weight": 0.35,
        "check": lambda claim, policy: _description_word_count(claim) < 10,
    },
    {
        "id": "FP-008",
        "name": "Single Document Only",
        "description": "High-value claim supported by only one document",
        "risk_weight": 0.35,
        "check": lambda claim, policy: (
            _is_high_value(claim) and _document_count(claim) <= 1
        ),
    },
    {
        "id": "FP-009",
        "name": "No Police Report for Major Incident",
        "description": "Major auto/vehicle incident without police report",
        "risk_weight": 0.4,
        "check": lambda claim, policy: (
            _is_auto_type(claim)
            and _is_high_value(claim)
            and not claim.get("police_report_number")
        ),
    },
    {
        "id": "FP-010",
        "name": "Claim Amount Near Policy Limit",
        "description": "Claimed amount is suspiciously close to coverage limit (>90%)",
        "risk_weight": 0.5,
        "check": lambda claim, policy: (
            policy.get("coverage_limit", 0) > 0
            and float(claim.get("estimated_amount", 0))
            > policy.get("coverage_limit", 0) * 0.90
        ),
    },
]


# ── US-Specific Patterns ─────────────────────────────────────────────────────

US_PATTERNS = [
    {
        "id": "FP-US-001",
        "name": "Auto Collision Without Witness",
        "description": "Single-vehicle auto collision with no witnesses or police report",
        "risk_weight": 0.45,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "auto_collision"
            and not claim.get("police_report_number")
        ),
    },
    {
        "id": "FP-US-002",
        "name": "Property Fire - Recent Policy Change",
        "description": "Property fire claim on policy started within 90 days",
        "risk_weight": 0.55,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "property_fire"
            and _days_after_start(claim, policy) <= 90
        ),
    },
    {
        "id": "FP-US-003",
        "name": "Medical Claim Spike",
        "description": "Medical claim amount is unusually high for the incident type",
        "risk_weight": 0.4,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "medical"
            and float(claim.get("estimated_amount", 0)) > 25_000
        ),
    },
    {
        "id": "FP-US-004",
        "name": "Liability Claim Without Third Party Info",
        "description": "Liability claim without police report or third-party documentation",
        "risk_weight": 0.45,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "liability"
            and not claim.get("police_report_number")
            and _document_count(claim) < 2
        ),
    },
]


# ── India-Specific Patterns ──────────────────────────────────────────────────

INDIA_PATTERNS = [
    {
        "id": "FP-IN-001",
        "name": "Own Damage Without FIR for High Value",
        "description": "High-value own damage claim without police FIR or panchnama",
        "risk_weight": 0.45,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "own_damage"
            and float(claim.get("estimated_amount", 0)) > 100_000
            and not claim.get("police_report_number")
        ),
    },
    {
        "id": "FP-IN-002",
        "name": "Theft Without FIR",
        "description": "Vehicle theft claim without police FIR number - FIR is mandatory for theft claims in India",
        "risk_weight": 0.8,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "theft"
            and not claim.get("police_report_number")
        ),
    },
    {
        "id": "FP-IN-003",
        "name": "Third Party Injury - No Medical Docs",
        "description": "Third-party injury claim with insufficient supporting documents",
        "risk_weight": 0.5,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") in ("third_party_injury", "third_party")
            and _document_count(claim) < 2
        ),
    },
    {
        "id": "FP-IN-004",
        "name": "Natural Calamity - No Area Verification",
        "description": "Natural calamity claim without location or weather verification docs",
        "risk_weight": 0.4,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "natural_calamity"
            and _document_count(claim) < 2
        ),
    },
    {
        "id": "FP-IN-005",
        "name": "Claim Exceeds IDV",
        "description": "Claimed amount exceeds the Insured Declared Value (IDV) of the vehicle",
        "risk_weight": 0.7,
        "check": lambda claim, policy: (
            float(claim.get("estimated_amount", 0)) > float(policy.get("idv", 0))
            if policy.get("idv")
            else False
        ),
    },
    {
        "id": "FP-IN-006",
        "name": "Own Damage After Policy Lapse",
        "description": "Own damage claim filed shortly after policy renewal from a lapsed state",
        "risk_weight": 0.55,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "own_damage"
            and policy.get("was_lapsed", False)
        ),
    },
]


# ── Statistical Baselines (for Anomaly Detector) ────────────────────────────

# US baselines (amounts in USD)
US_CLAIM_BASELINES = {
    "auto_collision": {
        "avg_amount": 6_500, "std_dev": 3_200,
        "median_amount": 5_800, "p95_amount": 18_000,
    },
    "auto_theft": {
        "avg_amount": 22_000, "std_dev": 12_000,
        "median_amount": 18_500, "p95_amount": 55_000,
    },
    "property_fire": {
        "avg_amount": 45_000, "std_dev": 28_000,
        "median_amount": 32_000, "p95_amount": 120_000,
    },
    "property_water": {
        "avg_amount": 12_000, "std_dev": 8_000,
        "median_amount": 9_500, "p95_amount": 35_000,
    },
    "liability": {
        "avg_amount": 18_000, "std_dev": 14_000,
        "median_amount": 12_000, "p95_amount": 75_000,
    },
    "medical": {
        "avg_amount": 8_500, "std_dev": 5_500,
        "median_amount": 6_200, "p95_amount": 28_000,
    },
}

# India baselines (amounts in INR)
INDIA_CLAIM_BASELINES = {
    "own_damage": {
        "avg_amount": 45_000, "std_dev": 30_000,
        "median_amount": 35_000, "p95_amount": 150_000,
    },
    "third_party_injury": {
        "avg_amount": 200_000, "std_dev": 150_000,
        "median_amount": 150_000, "p95_amount": 800_000,
    },
    "third_party_property": {
        "avg_amount": 50_000, "std_dev": 35_000,
        "median_amount": 40_000, "p95_amount": 175_000,
    },
    "third_party": {
        "avg_amount": 125_000, "std_dev": 100_000,
        "median_amount": 80_000, "p95_amount": 500_000,
    },
    "theft": {
        "avg_amount": 400_000, "std_dev": 250_000,
        "median_amount": 350_000, "p95_amount": 1_200_000,
    },
    "natural_calamity": {
        "avg_amount": 75_000, "std_dev": 50_000,
        "median_amount": 60_000, "p95_amount": 250_000,
    },
    "personal_accident": {
        "avg_amount": 300_000, "std_dev": 200_000,
        "median_amount": 200_000, "p95_amount": 1_000_000,
    },
    "fire": {
        "avg_amount": 150_000, "std_dev": 100_000,
        "median_amount": 120_000, "p95_amount": 500_000,
    },
}

# Backward-compatible alias — old code imports CLAIM_BASELINES directly
CLAIM_BASELINES = US_CLAIM_BASELINES


def _get_baselines() -> dict:
    """Return the country-appropriate baselines."""
    return INDIA_CLAIM_BASELINES if _is_india() else US_CLAIM_BASELINES


def _get_default_baseline() -> dict:
    """Fallback baseline when claim type is unknown."""
    if _is_india():
        return INDIA_CLAIM_BASELINES["own_damage"]
    return US_CLAIM_BASELINES["auto_collision"]


# ── Public API ───────────────────────────────────────────────────────────────

def get_patterns() -> list[dict]:
    """Return common + country-specific patterns for the active country."""
    if _is_india():
        return COMMON_PATTERNS + INDIA_PATTERNS
    return COMMON_PATTERNS + US_PATTERNS


def check_known_patterns(claim: dict, policy: dict) -> tuple[list[str], float]:
    """
    Run all applicable rule-based fraud patterns (common + country-specific).
    Returns (matched_pattern_names, composite_risk_score 0.0-1.0).
    """
    patterns = get_patterns()
    matched = []
    total_weight = 0.0
    max_possible = sum(p["risk_weight"] for p in patterns)

    for pattern in patterns:
        try:
            if pattern["check"](claim, policy):
                matched.append(f"{pattern['id']}: {pattern['name']} - {pattern['description']}")
                total_weight += pattern["risk_weight"]
        except Exception:
            continue

    risk_score = min(total_weight / max_possible if max_possible > 0 else 0, 1.0)
    return matched, risk_score


def get_statistical_anomaly(claim_type: str, amount: float) -> dict:
    """
    Check if a claim amount is statistically anomalous for its type.
    Uses country-appropriate baselines (USD for US, INR for India).
    Returns z-score and anomaly assessment.
    """
    baselines = _get_baselines()
    baseline = baselines.get(claim_type, _get_default_baseline())
    avg = baseline["avg_amount"]
    std = baseline["std_dev"]
    z_score = (amount - avg) / std if std > 0 else 0

    return {
        "z_score": round(z_score, 2),
        "average_for_type": avg,
        "claim_amount": amount,
        "is_outlier": abs(z_score) > 2.0,
        "is_extreme_outlier": abs(z_score) > 3.0,
        "percentile_estimate": "95th+" if amount > baseline["p95_amount"] else "normal range",
    }

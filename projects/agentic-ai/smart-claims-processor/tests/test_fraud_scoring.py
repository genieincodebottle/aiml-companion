"""Fraud scoring: country detection, pattern saturation, narrative parsing,
and the two-signal auto-reject rule.

Before these fixes:
  - `_is_india()` compared the country code "IN" against "india", so every
    India claim was scored with US patterns and US dollar baselines.
  - the pattern score divided by the weight of EVERY pattern in the
    database, so six matched red flags scored about 0.3.
  - the Consistency Validator's verdict was never read (consistency_score was
    hardcoded to 0.5), so a claimant confessing to fraud could not reach the
    auto-reject path.
"""

import pytest

from src.agents.fraud_crew import _crew_text, _parse_validation_score, score_fraud_signals
from src.config import set_country_override
from src.models.schemas import FraudRiskLevel
from src.tools import fraud_patterns as fp


@pytest.fixture
def india():
    set_country_override("india")
    yield
    set_country_override(None)


def test_india_profile_is_detected_from_its_iso_code(india):
    assert fp._active_country() == "IN"
    assert fp._is_india() is True


def test_us_profile_is_not_india():
    assert fp._is_india() is False


def test_india_amounts_are_compared_to_rupee_baselines(india):
    # 85,000 INR is an ordinary own-damage claim. Against the US dollar
    # baseline it was a z-score of about 24.
    result = fp.get_statistical_anomaly("own_damage", 85_000)
    assert not result["is_outlier"]


def test_india_uses_india_specific_patterns(india):
    ids = {p["id"] for p in fp.get_patterns()}
    assert "FP-IN-002" in ids and "FP-US-001" not in ids


def test_pattern_score_saturates_instead_of_diluting():
    claim = {"estimated_amount": 50_000, "incident_date": "2026-04-12", "incident_type": "auto_theft",
             "incident_description": "stolen", "documents": [], "police_report_number": ""}
    matched, score = fp.check_known_patterns(claim, {"start_date": "2020-01-01"})
    total = sum(p["risk_weight"] for p in fp.get_patterns()
                if any(m.startswith(p["id"] + ":") for m in matched))
    assert score == pytest.approx(min(total / 2.5, 1.0))


@pytest.mark.parametrize("text, expected", [
    ('{"story_consistent": false, "validation_score": 0.05, "analysis": "admits fraud"}', 0.05),
    ('```json\n{"validation_score": 0.8}\n```', 0.8),
    ('Here is my assessment: {"validation_score": 0.3, "analysis": "x"} Thanks.', 0.3),
    ('validation_score: 0.9 overall the story holds', 0.9),
    ("I could not assess this claim.", None),
    ("", None),
])
def test_validation_score_is_read_from_the_crew_output(text, expected):
    assert _parse_validation_score(_crew_text(text)) == expected


def test_confession_plus_strong_rules_is_confirmed_fraud():
    # Rules strongly suspicious, story judged fabricated, and the claimant's
    # own words state an intent to deceive.
    score, level, rec, conf = score_fraud_signals(0.86, 0.40, 0.97, fraud_admission=True)
    assert level == FraudRiskLevel.CONFIRMED and rec == "reject"
    assert score >= 0.75


def test_suspicion_without_an_admission_is_reviewed_not_rejected():
    # Live run: the US path B claim (new policy, keys in the ignition, no
    # police report) scored composite 0.835 and narrative risk 0.95 and was
    # auto-rejected. A suspicious story is not evidence.
    _, level, rec, _ = score_fraud_signals(0.94, 0.40, 0.95, fraud_admission=False)
    assert level == FraudRiskLevel.HIGH and rec == "escalate"
    _, level, _, _ = score_fraud_signals(0.94, 0.40, 0.95, fraud_admission=None)
    assert level != FraudRiskLevel.CONFIRMED


@pytest.mark.parametrize("text, expected", [
    ('{"validation_score": 0.0, "explicit_fraud_admission": true}', True),
    ('{"validation_score": 0.1, "explicit_fraud_admission": false}', False),
    ('{"validation_score": 0.1, "explicit_fraud_admission": "true"}', True),
    ('analysis ... explicit_fraud_admission: False', False),
    ('{"validation_score": 0.5}', None),
])
def test_admission_is_read_from_the_crew_output(text, expected):
    from src.agents.fraud_crew import _parse_admission
    assert _parse_admission(text) is expected


def test_suspicious_story_without_confession_goes_to_a_human_not_auto_reject():
    score, level, _, _ = score_fraud_signals(0.94, 0.40, 0.70)
    assert level in (FraudRiskLevel.HIGH, FraudRiskLevel.MEDIUM)
    assert score >= 0.45  # still pauses for review


def test_many_rule_hits_alone_never_auto_reject():
    # No narrative verdict: one signal family is not enough to reject.
    _, level, _, conf = score_fraud_signals(1.0, 0.7, None)
    assert level != FraudRiskLevel.CONFIRMED
    assert conf < 0.50  # below the fraud_crew gate, so a human looks


def test_confession_with_clean_rules_is_reviewed_not_auto_rejected():
    score, level, _, _ = score_fraud_signals(0.20, 0.15, 1.0)
    assert level != FraudRiskLevel.CONFIRMED
    assert score >= 0.45


def test_clean_claim_proceeds():
    _, level, rec, conf = score_fraud_signals(0.18, 0.15, 0.10)
    assert level == FraudRiskLevel.LOW and rec == "proceed"
    assert conf >= 0.5

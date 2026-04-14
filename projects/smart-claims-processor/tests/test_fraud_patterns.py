"""Tests for fraud pattern detection tools."""

import pytest
from src.tools.fraud_patterns import (
    check_known_patterns,
    get_statistical_anomaly,
    US_CLAIM_BASELINES as CLAIM_BASELINES,
)


def _make_claim(**kwargs):
    defaults = {
        "claim_id": "CLM-TEST",
        "incident_type": "auto_collision",
        "incident_date": "2024-11-15",
        "estimated_amount": 8500,
        "police_report_number": "APD-123",
    }
    defaults.update(kwargs)
    return defaults


def _make_policy(**kwargs):
    defaults = {
        "policy_number": "POL-001",
        "start_date": "2023-01-01",
        "end_date": "2025-01-01",
        "claims_count": 0,
        "coverage": {"collision": 50000},
    }
    defaults.update(kwargs)
    return defaults


def test_no_fraud_patterns_clean_claim():
    claim = _make_claim(estimated_amount=8500, incident_date="2024-11-15")
    policy = _make_policy()
    matched, score = check_known_patterns(claim, policy)
    assert score < 0.5, f"Expected low fraud score for clean claim, got {score}"


def test_round_number_flagged():
    claim = _make_claim(estimated_amount=10000)
    policy = _make_policy()
    matched, score = check_known_patterns(claim, policy)
    pattern_names = " ".join(matched)
    assert "Round Number" in pattern_names


def test_no_police_report_flagged():
    claim = _make_claim(incident_type="auto_collision", police_report_number=None)
    policy = _make_policy()
    matched, score = check_known_patterns(claim, policy)
    pattern_text = " ".join(matched).lower()
    assert "police report" in pattern_text or "without witness" in pattern_text


def test_repeat_claims_flagged():
    claim = _make_claim()
    policy = _make_policy(claims_count=2)
    matched, score = check_known_patterns(claim, policy)
    assert score > 0.20, "Repeat claimant should elevate fraud score"


def test_statistical_anomaly_normal():
    result = get_statistical_anomaly("auto_collision", 6500)
    assert not result["is_outlier"]
    assert abs(result["z_score"]) < 2.0


def test_statistical_anomaly_extreme():
    result = get_statistical_anomaly("auto_collision", 50000)
    assert result["is_outlier"] or result["is_extreme_outlier"]
    assert result["z_score"] > 2.0


def test_statistical_anomaly_known_types():
    for claim_type in CLAIM_BASELINES:
        result = get_statistical_anomaly(claim_type, CLAIM_BASELINES[claim_type]["avg_amount"])
        assert "z_score" in result
        assert abs(result["z_score"]) < 1.0, f"Average should not be anomalous for {claim_type}"


def test_score_bounded_0_1():
    claim = _make_claim(
        estimated_amount=50000,
        police_report_number=None,
        incident_date="2024-01-02",
    )
    policy = _make_policy(start_date="2024-01-01", claims_count=5)
    matched, score = check_known_patterns(claim, policy)
    assert 0.0 <= score <= 1.0

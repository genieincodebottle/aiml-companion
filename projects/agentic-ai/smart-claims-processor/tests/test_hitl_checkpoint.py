"""Tests for HITL checkpoint trigger logic."""

import pytest
from src.hitl.checkpoint import check_hitl_required
from src.models.schemas import FraudAssessmentOutput, FraudRiskLevel, HITLPriority, IntakeValidationOutput, ClaimType


def _make_intake(is_valid=True, flags=None):
    return IntakeValidationOutput(
        is_valid=is_valid,
        claim_type=ClaimType.AUTO_COLLISION,
        policy_active=True,
        claimant_eligible=True,
        missing_documents=[],
        intake_notes="OK",
        confidence=0.90,
        validation_flags=flags or [],
    )


def _make_fraud(score=0.20, risk=FraudRiskLevel.LOW):
    return FraudAssessmentOutput(
        fraud_risk_level=risk,
        fraud_score=score,
        primary_concerns=[],
        recommendation="proceed",
        crew_summary="Analysis complete",
        pattern_score=score,
        anomaly_score=score,
        consistency_score=0.90,
    )


def test_no_hitl_low_value_clean():
    claim = {"claim_id": "CLM-001", "policy_number": "POL-001", "estimated_amount": 2000, "is_appeal": False}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.10),
        damage_assessed_usd=2000,
        claim_history_count=0,
    )
    assert not req
    assert len(triggers) == 0


def test_hitl_high_value():
    claim = {"claim_id": "CLM-002", "policy_number": "POL-001", "estimated_amount": 15000, "is_appeal": False}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.15),
        damage_assessed_usd=15000,
    )
    assert req
    assert any("$15,000" in t or "threshold" in t.lower() for t in triggers)


def test_hitl_high_fraud():
    claim = {"claim_id": "CLM-003", "policy_number": "POL-001", "estimated_amount": 5000, "is_appeal": False}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.70, risk=FraudRiskLevel.HIGH),
        damage_assessed_usd=5000,
    )
    assert req
    assert any("fraud" in t.lower() for t in triggers)


def test_hitl_appeal_always():
    claim = {"claim_id": "CLM-004", "policy_number": "POL-001", "estimated_amount": 100, "is_appeal": True}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.10),
    )
    assert req
    assert any("appeal" in t.lower() for t in triggers)


def test_priority_critical_high_value():
    # Appeal always forces CRITICAL
    claim = {"claim_id": "CLM-005", "policy_number": "POL-001", "estimated_amount": 100000, "is_appeal": True}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.75, risk=FraudRiskLevel.HIGH),
        damage_assessed_usd=100000,
    )
    assert priority == HITLPriority.CRITICAL


def test_low_confidence_triggers_hitl():
    claim = {"claim_id": "CLM-006", "policy_number": "POL-001", "estimated_amount": 3000, "is_appeal": False}
    req, triggers, priority, score = check_hitl_required(
        claim=claim,
        intake_output=_make_intake(),
        fraud_output=_make_fraud(score=0.20),
        agent_confidence_scores=[0.40, 0.45],  # Below 0.65 threshold
    )
    assert req
    assert any("confidence" in t.lower() for t in triggers)

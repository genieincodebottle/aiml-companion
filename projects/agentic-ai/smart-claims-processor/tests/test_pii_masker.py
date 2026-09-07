"""Tests for PII masking layer."""

import pytest
from src.security.pii_masker import mask_claim, mask_text, get_masked_summary


def test_mask_email():
    assert "[EMAIL]" in mask_text("Contact jane.smith@gmail.com for details")
    assert "jane.smith@gmail.com" not in mask_text("Contact jane.smith@gmail.com")


def test_mask_phone():
    assert "[PHONE]" in mask_text("Call 555-123-4567 now")
    assert "555-123-4567" not in mask_text("Call 555-123-4567 now")


def test_mask_ssn():
    assert "[SSN]" in mask_text("SSN: 123-45-6789")


def test_mask_name_field():
    claim = {"claimant_name": "Jane Smith", "incident_type": "auto_collision"}
    masked = mask_claim(claim)
    assert masked["claimant_name"] == "[CLAIMANT_NAME]"
    assert masked["incident_type"] == "auto_collision"  # Non-PII unchanged


def test_redact_dob():
    claim = {"claimant_dob": "1985-03-15"}
    masked = mask_claim(claim)
    assert masked["claimant_dob"] == "[REDACTED]"


def test_mask_preserves_structure():
    claim = {
        "claim_id": "CLM-001",
        "claimant_name": "Jane Smith",
        "claimant_email": "jane@email.com",
        "estimated_amount": 8500,
    }
    masked = mask_claim(claim)
    assert masked["claim_id"] == "CLM-001"        # Non-PII unchanged
    assert masked["estimated_amount"] == 8500      # Numbers unchanged
    assert masked["claimant_name"] == "[CLAIMANT_NAME]"
    assert "[EMAIL]" in masked["claimant_email"]


def test_masked_summary_no_pii():
    claim = {
        "claim_id": "CLM-001",
        "policy_number": "POL-123",
        "claimant_name": "Jane Smith",
        "incident_type": "auto_collision",
        "incident_date": "2024-11-15",
        "incident_location": "Austin, TX",
        "estimated_amount": 8500,
        "incident_description": "Rear-ended, contact jane@email.com",
    }
    summary = get_masked_summary(claim)
    assert "Jane Smith" not in summary
    assert "jane@email.com" not in summary
    assert "CLM-001" in summary
    assert "8,500" in summary


def test_mask_nested_dict():
    claim = {
        "claimant": {
            "name": "Bob Jones",
            "contact": {"email": "bob@test.com", "phone": "555-000-1111"},
        }
    }
    masked = mask_claim(claim)
    inner = masked["claimant"]["contact"]
    assert "[EMAIL]" in inner["email"]
    assert "[PHONE]" in inner["phone"]

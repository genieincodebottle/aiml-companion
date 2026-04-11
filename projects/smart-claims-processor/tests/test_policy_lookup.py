"""Tests for policy lookup tools."""

import pytest
from src.tools.policy_lookup import (
    lookup_policy,
    is_policy_active,
    get_coverage_for_claim_type,
)


def test_lookup_existing_policy():
    policy = lookup_policy("POL-AUTO-789456")
    assert policy is not None
    assert policy["policy_number"] == "POL-AUTO-789456"
    assert policy["type"] == "auto"


def test_lookup_nonexistent_policy():
    policy = lookup_policy("POL-DOES-NOT-EXIST")
    assert policy is None


def test_policy_active_on_incident_date():
    policy = lookup_policy("POL-AUTO-789456")
    active, reason = is_policy_active(policy, "2024-06-15")
    assert active is True


def test_lapsed_policy():
    policy = lookup_policy("POL-AUTO-112233")
    assert policy["status"] == "lapsed"
    active, reason = is_policy_active(policy, "2024-08-15")
    assert active is False
    assert "lapsed" in reason.lower() or "inactive" in reason.lower() or "status" in reason.lower()


def test_incident_before_policy_start():
    policy = lookup_policy("POL-AUTO-789456")
    active, reason = is_policy_active(policy, "2022-01-01")
    assert active is False


def test_coverage_auto_collision():
    policy = lookup_policy("POL-AUTO-789456")
    coverage = get_coverage_for_claim_type(policy, "auto_collision")
    assert coverage["covered"] is True
    assert coverage["coverage_limit"] > 0
    assert coverage["deductible"] >= 0


def test_coverage_not_found_for_type():
    policy = lookup_policy("POL-AUTO-789456")
    coverage = get_coverage_for_claim_type(policy, "property_fire")
    # Auto policy shouldn't cover property fire
    assert coverage["coverage_limit"] == 0 or coverage["covered"] is False


def test_coverage_net_max_payout():
    policy = lookup_policy("POL-AUTO-789456")
    coverage = get_coverage_for_claim_type(policy, "auto_collision")
    expected_net = max(0, coverage["coverage_limit"] - coverage["deductible"])
    assert coverage["net_max_payout"] == expected_net

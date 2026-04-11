"""Tests for damage calculation tools."""

import pytest
from src.tools.damage_calculator import (
    calculate_vehicle_acv,
    should_total_loss,
    apply_depreciation,
)


def test_new_vehicle_high_acv():
    acv = calculate_vehicle_acv(2024, "Toyota", "Camry")
    assert acv > 15000
    assert acv < 50000


def test_old_vehicle_low_acv():
    acv = calculate_vehicle_acv(2010, "Ford", "Focus")
    assert acv < 15000
    assert acv > 1000


def test_acv_decreases_with_age():
    newer = calculate_vehicle_acv(2022, "Honda", "Civic")
    older = calculate_vehicle_acv(2015, "Honda", "Civic")
    assert newer > older


def test_total_loss_triggered():
    # Repair cost = $30K, ACV = $35K -> ratio = 0.857 > 0.75 -> total loss
    is_total, ratio = should_total_loss(30000, 35000)
    assert is_total is True
    assert ratio > 0.75


def test_no_total_loss():
    # Repair cost = $5K, ACV = $25K -> ratio = 0.20 < 0.75 -> repair
    is_total, ratio = should_total_loss(5000, 25000)
    assert is_total is False
    assert ratio < 0.75


def test_zero_acv_no_total_loss():
    is_total, ratio = should_total_loss(1000, 0)
    assert is_total is False


def test_depreciation_applied():
    depreciated, depr_amount = apply_depreciation(10000, "auto", 3)
    assert depreciated < 10000
    assert depr_amount > 0
    assert depreciated + depr_amount <= 10000 * 1.01  # Within rounding error


def test_no_depreciation_new_vehicle():
    # Age 0 - minimum depreciation
    depreciated, depr_amount = apply_depreciation(10000, "auto", 0)
    assert depreciated == 10000  # Age 0 = no depreciation
    assert depr_amount == 0


def test_max_depreciation_cap():
    # Very old vehicle - depreciation should never exceed 70%
    depreciated, depr_amount = apply_depreciation(10000, "auto", 30)
    assert depreciated >= 3000  # At least 30% remains
    assert depr_amount <= 7000  # Max 70% depreciation

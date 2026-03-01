"""Tests for drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitor import check_data_quality, classify_psi, compute_psi, monitor_drift


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputePSI:
    """Tests for compute_psi()."""

    def test_identical_distributions(self):
        """PSI of identical distributions should be near zero."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distribution(self):
        """PSI of shifted distributions should be > 0."""
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)  # Shifted by 2 std devs
        psi = compute_psi(ref, cur)
        assert psi > 0.1

    def test_handles_empty(self):
        """PSI of empty arrays should return 0."""
        psi = compute_psi(np.array([]), np.array([1, 2, 3]))
        assert psi == 0.0

    def test_handles_nan(self):
        """PSI should ignore NaN values."""
        ref = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 100)
        cur = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        psi = compute_psi(ref, cur)
        assert psi >= 0


class TestClassifyPSI:
    """Tests for classify_psi()."""

    def test_stable(self):
        assert classify_psi(0.05) == "stable"

    def test_moderate(self):
        assert classify_psi(0.15) == "moderate_shift"

    def test_significant(self):
        assert classify_psi(0.30) == "significant_shift"


class TestMonitorDrift:
    """Tests for monitor_drift()."""

    def test_returns_dict(self):
        np.random.seed(42)
        ref = pd.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.normal(5, 2, 500)})
        cur = pd.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.normal(5, 2, 500)})
        result = monitor_drift(ref, cur)
        assert isinstance(result, dict)
        assert "a" in result
        assert "psi" in result["a"]

    def test_detects_drift(self):
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": np.random.normal(3, 1, 500)})
        result = monitor_drift(ref, cur)
        assert result["x"]["status"] != "stable"


class TestCheckDataQuality:
    """Tests for check_data_quality()."""

    def test_clean_data_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_data_quality(df)
        assert result["pass"] is True
        assert result["n_issues"] == 0

    def test_detects_high_missing(self):
        df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, 5]})
        result = check_data_quality(df)
        assert result["n_issues"] > 0
        assert any(i["type"] == "missing_values" for i in result["issues"])

    def test_detects_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        result = check_data_quality(df)
        assert any(i["type"] == "duplicates" for i in result["issues"])

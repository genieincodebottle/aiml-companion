"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_age_buckets,
    add_dti_ratio,
    add_loan_burden,
    add_log_transforms,
    add_utilization,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create sample DataFrame with credit risk features."""
    return pd.DataFrame({
        "duration": [12, 24, 36, 48, 6],
        "credit_amount": [1000, 5000, 10000, 20000, 500],
        "age": [22, 35, 45, 55, 65],
        "income": [30000, 50000, 70000, 90000, 25000],
        "existing_credits": [1, 2, 3, 4, 1],
        "balance": [500, 2000, 5000, 15000, 200],
        "credit_limit": [2000, 5000, 10000, 20000, 1000],
        "target": [0, 0, 1, 1, 0],
    })


@pytest.fixture
def cfg():
    """Config for feature engineering."""
    return {
        "dti_columns": {"debt": "credit_amount", "income": "income"},
        "utilization_columns": {"balance": "balance", "limit": "credit_limit"},
        "burden_columns": {"amount": "credit_amount", "duration": "duration", "income": "income"},
        "age_column": "age",
        "skew_threshold": 2.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDTIRatio:
    """Tests for add_dti_ratio()."""

    def test_creates_dti_column(self, sample_df, cfg):
        result = add_dti_ratio(sample_df, cfg)
        assert "dti_ratio" in result.columns

    def test_dti_capped_at_one(self, sample_df, cfg):
        result = add_dti_ratio(sample_df, cfg)
        assert result["dti_ratio"].max() <= 1.0

    def test_does_not_modify_input(self, sample_df, cfg):
        original_cols = list(sample_df.columns)
        add_dti_ratio(sample_df, cfg)
        assert list(sample_df.columns) == original_cols


class TestUtilization:
    """Tests for add_utilization()."""

    def test_creates_utilization_column(self, sample_df, cfg):
        result = add_utilization(sample_df, cfg)
        assert "utilization_ratio" in result.columns

    def test_utilization_range(self, sample_df, cfg):
        result = add_utilization(sample_df, cfg)
        assert result["utilization_ratio"].min() >= 0
        assert result["utilization_ratio"].max() <= 1.5


class TestLoanBurden:
    """Tests for add_loan_burden()."""

    def test_creates_burden_column(self, sample_df, cfg):
        result = add_loan_burden(sample_df, cfg)
        assert "loan_burden" in result.columns

    def test_burden_positive(self, sample_df, cfg):
        result = add_loan_burden(sample_df, cfg)
        assert (result["loan_burden"] >= 0).all()


class TestAgeBuckets:
    """Tests for add_age_buckets()."""

    def test_creates_age_group(self, sample_df, cfg):
        result = add_age_buckets(sample_df, cfg)
        assert "age_group" in result.columns

    def test_correct_bucket_assignment(self, sample_df, cfg):
        result = add_age_buckets(sample_df, cfg)
        # age=22 should be "18-25"
        assert result.iloc[0]["age_group"] == "18-25"
        # age=35 should be "26-35"
        assert result.iloc[1]["age_group"] == "26-35"


class TestLogTransforms:
    """Tests for add_log_transforms()."""

    def test_adds_log_columns_for_skewed(self, cfg):
        # Create highly skewed data
        np.random.seed(42)
        df = pd.DataFrame({
            "credit_amount": np.random.exponential(5000, 100),
            "duration": np.random.normal(24, 6, 100),
            "target": np.random.choice([0, 1], 100),
        })
        result = add_log_transforms(df, cfg)
        # credit_amount is exponential (skewed), should get log transform
        log_cols = [c for c in result.columns if c.startswith("log_")]
        assert len(log_cols) >= 1


class TestEngineerFeatures:
    """Tests for the full engineer_features() pipeline."""

    def test_adds_multiple_features(self, sample_df, cfg):
        result = engineer_features(sample_df, cfg)
        assert result.shape[1] > sample_df.shape[1]

    def test_preserves_original_columns(self, sample_df, cfg):
        result = engineer_features(sample_df, cfg)
        for col in sample_df.columns:
            assert col in result.columns

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
        # Create highly skewed data. Lognormal with sigma=2 has sample skew
        # well above the 2.0 threshold (exponential hovers near it and can
        # fall below for small samples).
        np.random.seed(42)
        df = pd.DataFrame({
            "credit_amount": np.random.lognormal(8, 2, 100),
            "duration": np.random.normal(24, 6, 100),
            "target": np.random.choice([0, 1], 100),
        })
        result = add_log_transforms(df, cfg)
        # credit_amount is lognormal (heavily skewed), should get log transform
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


def test_protected_attributes_never_reach_the_model():
    """A credit model may not learn from sex or marital status.

    `personal_status` in the German Credit data reads "male single", "female
    div/dep/mar" -- it encodes sex, and it used to go into the model as an
    ordinary categorical. ECOA / Regulation B prohibits that. Measured cost of
    removing it: 0.005 AUC. Measured cost of keeping it: female default rate
    35.2% against 27.7% for men, learned and applied.
    """
    import pandas as pd
    from src.features import engineer_features, PROTECTED_COLUMNS

    df = pd.DataFrame({
        "personal_status": ["male single", "female div/dep/mar"] * 10,
        "age": list(range(20, 40)),
        "credit_amount": list(range(1000, 3000, 100)),
        "duration": [12] * 20,
        "target": [0, 1] * 10,
    })
    out = engineer_features(df, {})
    assert "personal_status" not in out.columns
    for col in out.columns:
        assert col.lower() not in {p.lower() for p in PROTECTED_COLUMNS}


def test_features_are_skipped_not_faked_when_inputs_are_missing():
    """A feature that cannot be computed must be absent, not silently null.

    dti_ratio used to resolve `income` to `personal_status` -- a categorical --
    producing a 100%-NaN column that logged success and got imputed downstream.
    A trustworthy name on an empty column is worse than no column.
    """
    import pandas as pd
    from src.features import engineer_features

    df = pd.DataFrame({           # no income, no balance, no credit limit
        "age": list(range(20, 40)),
        "credit_amount": list(range(1000, 3000, 100)),
        "duration": [12] * 20,
        "target": [0, 1] * 10,
    })
    out = engineer_features(df, {})

    assert "dti_ratio" not in out.columns, "DTI is undefined without income"
    assert "utilization_ratio" not in out.columns, "needs a balance and a limit"
    # anything that IS created must carry real values
    for col in set(out.columns) - set(df.columns):
        if out[col].dtype.name not in ("category", "object"):
            assert out[col].notna().any(), f"{col} is entirely null"

"""Tests for data loading and cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_data, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a sample credit risk DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "duration": np.random.randint(6, 48, n),
        "credit_amount": np.random.randint(500, 20000, n),
        "age": np.random.randint(19, 75, n),
        "income": np.random.randint(15000, 80000, n),
        "existing_credits": np.random.randint(1, 5, n),
        "housing": np.random.choice(["own", "rent", "free"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def cfg():
    """Minimal config for testing."""
    return {
        "target_column": "target",
        "numeric_columns": ["duration", "credit_amount", "age", "income"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanData:
    """Tests for clean_data()."""

    def test_removes_duplicates(self, sample_df, cfg):
        """Exact duplicate rows should be removed."""
        df_with_dupes = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        cleaned = clean_data(df_with_dupes, cfg)
        assert len(cleaned) == len(sample_df)

    def test_preserves_target(self, sample_df, cfg):
        """Target column must survive cleaning."""
        cleaned = clean_data(sample_df, cfg)
        assert "target" in cleaned.columns

    def test_snake_case_columns(self, cfg):
        """Column names should be converted to snake_case."""
        df = pd.DataFrame({
            "CreditAmount": [1000],
            "LoanDuration": [12],
            "target": [0],
        })
        cleaned = clean_data(df, cfg)
        assert "credit_amount" in cleaned.columns
        assert "loan_duration" in cleaned.columns

    def test_raises_on_missing_target(self, sample_df, cfg):
        """Should raise ValueError if target column is missing."""
        df = sample_df.drop(columns=["target"])
        with pytest.raises(ValueError, match="Target column"):
            clean_data(df, cfg)

    def test_coerces_numeric(self, cfg):
        """Non-numeric values in numeric columns should become NaN."""
        df = pd.DataFrame({
            "duration": ["12", "abc", "24"],
            "credit_amount": [1000, 2000, 3000],
            "target": [0, 1, 0],
        })
        cleaned = clean_data(df, cfg)
        assert cleaned["duration"].isna().sum() == 1

    def test_output_is_copy(self, sample_df, cfg):
        """Cleaning should not modify the original DataFrame."""
        original_len = len(sample_df)
        clean_data(sample_df, cfg)
        assert len(sample_df) == original_len


class TestLoadConfig:
    """Tests for load_config()."""

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for non-existent config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

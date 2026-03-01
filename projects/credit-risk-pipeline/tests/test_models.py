"""Tests for model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import build_gbc_pipeline, build_lr_pipeline, build_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "duration": np.random.randint(6, 48, n),
        "credit_amount": np.random.randint(500, 20000, n).astype(float),
        "age": np.random.randint(19, 75, n).astype(float),
        "income": np.random.randint(15000, 80000, n).astype(float),
        "housing": np.random.choice(["own", "rent", "free"], n),
        "purpose": np.random.choice(["car", "education", "business"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    X = df.drop(columns=["target"])
    y = df["target"]
    numeric_cols = ["duration", "credit_amount", "age", "income"]
    categorical_cols = ["housing", "purpose"]
    return X, y, numeric_cols, categorical_cols


@pytest.fixture
def cfg():
    return {
        "knn_imputer_neighbors": 3,
        "logistic_regression": {"C": 1.0, "max_iter": 500},
        "gradient_boosting": {"n_estimators": 50, "max_depth": 3},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    """Tests for build_preprocessor()."""

    def test_transforms_all_columns(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        preprocessor = build_preprocessor(num_cols, cat_cols, cfg)
        preprocessor.fit(X)
        X_t = preprocessor.transform(X)
        # Should have more columns than input (one-hot encoding)
        assert X_t.shape[1] > len(num_cols)

    def test_handles_missing_values(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        # Introduce NaN
        X_with_nan = X.copy()
        X_with_nan.loc[0, "age"] = np.nan
        X_with_nan.loc[1, "income"] = np.nan

        preprocessor = build_preprocessor(num_cols, cat_cols, cfg)
        preprocessor.fit(X_with_nan)
        X_t = preprocessor.transform(X_with_nan)
        # Should have no NaN after KNN imputation
        assert not np.isnan(X_t).any()


class TestLogisticRegression:
    """Tests for build_lr_pipeline()."""

    def test_fits_and_predicts(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        pipeline = build_lr_pipeline(num_cols, cat_cols, cfg)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        pipeline = build_lr_pipeline(num_cols, cat_cols, cfg)
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestGradientBoosting:
    """Tests for build_gbc_pipeline()."""

    def test_fits_and_predicts(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        pipeline = build_gbc_pipeline(num_cols, cat_cols, cfg)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba(self, sample_data, cfg):
        X, y, num_cols, cat_cols = sample_data
        pipeline = build_gbc_pipeline(num_cols, cat_cols, cfg)
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(y), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

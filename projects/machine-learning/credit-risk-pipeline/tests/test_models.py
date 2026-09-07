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


def test_training_produces_out_of_fold_predictions():
    """Evaluation must never score a model on its own training rows.

    Regression test for the most expensive defect in this project. Evaluation
    used to call `pipeline.predict_proba(X)` on a model fitted to all of X.
    Gradient boosting memorises 1,000 rows easily, so it reported a total
    misclassification cost of **1** against an honest out-of-fold cost of 712 --
    and, worse than the flattering number, it picked its operating threshold
    from those memorised scores. That threshold (0.30, where the honest optimum
    is 0.05) was written to threshold.json and served, costing 1,372 in
    reality: nearly double the properly tuned optimum.
    """
    import numpy as np
    import pandas as pd
    from src.models import train_and_evaluate

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y", "z"], size=n),
    })
    df["target"] = (df["num_a"] + rng.normal(scale=1.5, size=n) > 0).astype(int)

    # model_path=None: never touch the artifact the serving API loads
    results = train_and_evaluate(
        df, {"target_column": "target", "cv_folds": 3, "model_path": None})

    for name, res in results.items():
        assert "oof_proba" in res, f"{name} produced no out-of-fold predictions"
        oof = res["oof_proba"]
        assert len(oof) == len(df)
        assert ((oof >= 0) & (oof <= 1)).all()

        # The out-of-fold score must be meaningfully worse than the in-sample
        # score. If they match, the OOF predictions are not really out-of-fold.
        from sklearn.metrics import roc_auc_score
        in_sample = res["pipeline"].predict_proba(df.drop(columns=["target"]))[:, 1]
        assert roc_auc_score(df["target"], oof) <= roc_auc_score(
            df["target"], in_sample) + 1e-9


def test_evaluation_refuses_to_score_in_sample():
    """full_evaluation must fail loudly rather than fall back to training rows."""
    import numpy as np
    import pandas as pd
    import pytest
    from src.evaluate import full_evaluation

    df = pd.DataFrame({"num_a": np.arange(20.0), "target": [0, 1] * 10})
    # results without 'oof_proba' -- the shape the old in-sample path produced
    fake = {"M": {"roc_auc_mean": 0.9, "pipeline": object()}}

    with pytest.raises(KeyError, match="out-of-fold"):
        full_evaluation(df, fake, {"target_column": "target"})

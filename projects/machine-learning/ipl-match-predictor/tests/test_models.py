"""
test_models.py - Unit Tests for ML Model Pipelines
=====================================================

Run with::

    pytest tests/test_models.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models import (
    build_classifier,
    build_regressor,
    build_xgb_classifier,
    build_neural_classifier,
    cross_validate_model,
    evaluate_model,
    get_feature_importances,
    split_data,
    _HAS_XGBOOST,
    _HAS_TORCH,
)


@pytest.fixture()
def synthetic_classification_data():
    """Generate synthetic binary classification data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
    })
    y = pd.Series((X["feature_1"] + X["feature_2"] > 0).astype(int))
    return X, y


@pytest.fixture()
def synthetic_regression_data():
    """Generate synthetic regression data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
    })
    y = pd.Series(3 * X["feature_1"] + 2 * X["feature_2"] + np.random.randn(n) * 0.5)
    return X, y


class TestBuildClassifier:
    """Tests for the Random Forest classifier builder."""

    def test_returns_pipeline(self):
        clf = build_classifier()
        assert isinstance(clf, Pipeline)

    def test_has_scaler_and_classifier(self):
        clf = build_classifier()
        step_names = [name for name, _ in clf.steps]
        assert "scaler" in step_names
        assert "classifier" in step_names

    def test_custom_params(self):
        clf = build_classifier(n_estimators=50, max_depth=5)
        rf = clf.named_steps["classifier"]
        assert rf.n_estimators == 50
        assert rf.max_depth == 5


class TestBuildRegressor:
    """Tests for the Gradient Boosting regressor builder."""

    def test_returns_pipeline(self):
        reg = build_regressor()
        assert isinstance(reg, Pipeline)

    def test_custom_learning_rate(self):
        reg = build_regressor(learning_rate=0.05)
        gb = reg.named_steps["regressor"]
        assert gb.learning_rate == 0.05


class TestSplitData:
    """Tests for the train/test split function."""

    def test_split_sizes(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
        assert len(X_train) == pytest.approx(150, abs=5)
        assert len(X_test) == pytest.approx(50, abs=5)

    def test_preserves_features(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, _, _ = split_data(X, y)
        assert list(X_train.columns) == list(X.columns)


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_classification_metrics(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        clf = build_classifier(n_estimators=10)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, task="classification")
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_regression_metrics(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        X_train, X_test, y_train, y_test = split_data(X, y, stratify=False)
        reg = build_regressor(n_estimators=10)
        reg.fit(X_train, y_train)
        metrics = evaluate_model(reg, X_test, y_test, task="regression")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0


class TestCrossValidate:
    """Tests for cross-validation."""

    def test_returns_mean_and_std(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        clf = build_classifier(n_estimators=10)
        result = cross_validate_model(clf, X, y, cv=3)
        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert len(result["scores"]) == 3


class TestFeatureImportances:
    """Tests for feature importance extraction."""

    def test_returns_sorted(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        clf = build_classifier(n_estimators=10)
        clf.fit(X, y)
        fi = get_feature_importances(clf, list(X.columns), top_n=3)
        assert len(fi) == 3
        assert fi["Importance"].is_monotonic_decreasing


@pytest.mark.skipif(not _HAS_XGBOOST, reason="xgboost not installed")
class TestBuildXGBClassifier:
    """Tests for the XGBoost classifier builder."""

    def test_returns_pipeline(self):
        clf = build_xgb_classifier()
        assert isinstance(clf, Pipeline)

    def test_has_scaler_and_classifier(self):
        clf = build_xgb_classifier()
        step_names = [name for name, _ in clf.steps]
        assert "scaler" in step_names
        assert "classifier" in step_names

    def test_custom_params(self):
        clf = build_xgb_classifier(n_estimators=50, max_depth=4, learning_rate=0.05)
        xgb = clf.named_steps["classifier"]
        assert xgb.n_estimators == 50
        assert xgb.max_depth == 4
        assert xgb.learning_rate == 0.05

    def test_fit_and_predict(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        clf = build_xgb_classifier(n_estimators=10)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, task="classification")
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["accuracy"] > 0.5  # better than random on easy data

    def test_feature_importances(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        clf = build_xgb_classifier(n_estimators=10)
        clf.fit(X, y)
        fi = get_feature_importances(clf, list(X.columns), top_n=3)
        assert len(fi) == 3

    def test_cross_validation(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        clf = build_xgb_classifier(n_estimators=10)
        result = cross_validate_model(clf, X, y, cv=3)
        assert result["mean"] > 0.5


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestBuildNeuralClassifier:
    """Tests for the Neural Network classifier builder."""

    def test_returns_pipeline(self):
        clf = build_neural_classifier()
        assert isinstance(clf, Pipeline)

    def test_has_scaler_and_classifier(self):
        clf = build_neural_classifier()
        step_names = [name for name, _ in clf.steps]
        assert "scaler" in step_names
        assert "classifier" in step_names

    def test_fit_and_predict(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        clf = build_neural_classifier(epochs=20, batch_size=32)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        clf = build_neural_classifier(epochs=20)
        clf.fit(X_train, y_train)
        nn_clf = clf.named_steps["classifier"]
        probs = nn_clf.predict_proba(X_test)
        assert probs.shape == (len(X_test), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_accuracy_above_random(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        X_train, X_test, y_train, y_test = split_data(X, y)
        clf = build_neural_classifier(epochs=50)
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, task="classification")
        assert metrics["accuracy"] > 0.5

    def test_cross_validation(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        clf = build_neural_classifier(epochs=20)
        result = cross_validate_model(clf, X, y, cv=3)
        assert "mean" in result
        assert len(result["scores"]) == 3

def test_season_split_never_trains_on_the_future():
    """The test seasons must sit entirely after the training seasons.

    This project predicts matches that have not been played. A random split
    trains on 2025 to predict 2024, which deployment cannot do, and it flatters
    the model: 0.5150 random against 0.4507 on an honest 2025 holdout.
    """
    import numpy as np
    import pandas as pd
    from src.models import split_data_by_season

    X = pd.DataFrame({"f": range(100)})
    y = pd.Series([0, 1] * 50)
    seasons = np.repeat([2019, 2020, 2021, 2022, 2023], 20)

    X_tr, X_te, y_tr, y_te = split_data_by_season(X, y, seasons, holdout_seasons=2)

    train_seasons = seasons[X_tr.index]
    test_seasons = seasons[X_te.index]
    assert train_seasons.max() < test_seasons.min(), "a train season is in the future"
    assert set(test_seasons) == {2022, 2023}
    assert len(X_tr) + len(X_te) == len(X)


def test_season_split_refuses_impossible_holdouts():
    """Better to fail than to silently return an empty training set."""
    import numpy as np
    import pandas as pd
    import pytest
    from src.models import split_data_by_season

    X = pd.DataFrame({"f": range(20)})
    y = pd.Series([0, 1] * 10)
    seasons = np.repeat([2022, 2023], 10)

    with pytest.raises(ValueError, match="seasons"):
        split_data_by_season(X, y, seasons, holdout_seasons=2)


def test_majority_baseline_is_reported_alongside_accuracy():
    """Accuracy on a near-balanced binary problem is unreadable alone.

    Always predicting the same team scores ~0.52 on this data, so a model at
    0.53 has demonstrated nothing. The evaluation report must carry both.
    """
    import numpy as np
    from src.models import majority_class_baseline

    # training set leans to class 1 -> baseline predicts 1 everywhere
    y_train = np.array([1] * 70 + [0] * 30)
    y_test = np.array([1] * 60 + [0] * 40)
    assert majority_class_baseline(y_train, y_test) == 0.6

    y_train = np.array([0] * 80 + [1] * 20)
    assert majority_class_baseline(y_train, y_test) == 0.4


def test_report_flags_a_model_that_loses_to_the_baseline():
    from src.evaluate import generate_report

    report = generate_report({
        "classification": {"accuracy": 0.4507, "f1": 0.4268,
                           "baseline_accuracy": 0.4789},
        "best_model": "Random Forest",
    })
    assert "Majority-class baseline" in report
    assert "does NOT beat the baseline" in report

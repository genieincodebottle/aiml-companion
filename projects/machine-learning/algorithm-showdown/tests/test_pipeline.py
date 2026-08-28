"""
Tests for the ML Algorithms Capstone pipeline.

Run: pytest tests/ -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def test_data_loading():
    """Test that breast cancer dataset loads correctly."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert set(np.unique(y)) == {0, 1}


def test_build_algorithms():
    """Test that all 6 algorithms are built correctly."""
    from train import build_algorithms
    algorithms = build_algorithms()
    assert len(algorithms) == 6
    expected_names = [
        'Logistic Regression', 'SVM (RBF)', 'KNN (k=5)',
        'Decision Tree', 'Random Forest', 'XGBoost'
    ]
    for name in expected_names:
        assert name in algorithms


def test_pipeline_fit_predict():
    """Test that each pipeline can fit and predict."""
    from train import build_algorithms
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    algorithms = build_algorithms()
    for name, pipeline in algorithms.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert set(np.unique(predictions)).issubset({0, 1})
        accuracy = (predictions == y_test).mean()
        assert accuracy > 0.8


def test_threshold_tuning():
    """Test that threshold tuning returns valid values."""
    from train import build_algorithms, tune_threshold
    data = load_breast_cancer()
    X, y = data.data, data.target

    algorithms = build_algorithms()
    best_pipe, threshold = tune_threshold(X, y, algorithms, target_recall=0.99)

    assert best_pipe is not None
    assert 0.0 < threshold < 1.0


def test_prediction_probabilities():
    """Test that predict_proba returns valid probabilities."""
    from train import build_algorithms
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    algorithms = build_algorithms()
    for name, pipeline in algorithms.items():
        pipeline.fit(X_train, y_train)
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(X_test)
            assert proba.shape == (len(X_test), 2)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

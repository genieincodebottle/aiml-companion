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
    from src.train import build_algorithms
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
    from src.train import build_algorithms
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
    from src.train import build_algorithms, tune_threshold
    data = load_breast_cancer()
    X, y = data.data, data.target

    algorithms = build_algorithms()
    best_pipe, threshold = tune_threshold(X, y, algorithms, target_recall=0.99)

    assert best_pipe is not None
    assert 0.0 < threshold < 1.0


def test_prediction_probabilities():
    """Test that predict_proba returns valid probabilities."""
    from src.train import build_algorithms
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


def test_splits_are_three_way_and_disjoint():
    """Threshold tuning is model selection, so it needs its own split.

    Regression test for a real defect: the pipeline used a single train/test
    split, tuned the decision threshold on the test set, and then reported
    performance on that same test set. The reported recall was therefore the
    target restated -- across 30 seeds it had a standard deviation of 0.0005,
    because the threshold was *defined* as the one hitting that recall there.
    """
    from sklearn.datasets import load_breast_cancer
    from src.train import make_splits

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y)

    assert len(X_tr) + len(X_va) + len(X_te) == len(X)
    # rows must not be shared between the three roles
    rows = lambda A: {tuple(r) for r in A}
    assert not (rows(X_tr) & rows(X_va))
    assert not (rows(X_tr) & rows(X_te))
    assert not (rows(X_va) & rows(X_te)), "test set overlaps the tuning set"
    # every split must contain both classes, or stratification silently failed
    for part in (y_tr, y_va, y_te):
        assert set(np.unique(part)) == {0, 1}


def test_evaluate_scores_the_untouched_test_split():
    """evaluate.py must use train.py's split, not invent a coinciding one."""
    from sklearn.datasets import load_breast_cancer
    from src.train import make_splits
    import src.evaluate as ev
    import inspect

    src_text = inspect.getsource(ev)
    assert "make_splits" in src_text, (
        "evaluate.py must import the shared split definition rather than "
        "rebuilding one that only happens to agree")

    X, y = load_breast_cancer(return_X_y=True)
    _, _, X_te, _, _, y_te = make_splits(X, y)
    assert len(X_te) == len(y_te) and len(X_te) > 0


def test_threshold_comes_from_the_curve_not_a_silent_default():
    """pick_threshold must return a real threshold and keep its indices aligned.

    The old code did `thresholds[idx] if idx < len(thresholds) else 0.5`, which
    shipped an untuned 0.5 as though it had been tuned whenever the index landed
    on the final (recall=0) point that has no threshold behind it.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from src.train import build_algorithms, pick_threshold

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)
    pipe = build_algorithms()['Logistic Regression'].fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 0]

    for target in (0.80, 0.90, 0.95, 0.99):
        thr, prec, rec = pick_threshold(y_te, proba, target)
        assert 0.0 <= thr <= 1.0
        assert 0.0 <= prec <= 1.0 and 0.0 <= rec <= 1.0
        # the returned recall should actually be near what was asked for
        assert abs(rec - target) < 0.10, f"target {target} -> recall {rec}"


def test_selection_uses_the_comparison_it_just_ran():
    """The bake-off must choose the model, not be decorative.

    select_best used to be a hardcoded 'XGBoost', so six algorithms were
    compared and the result was then ignored.
    """
    from src.train import select_best

    results = {
        'Logistic Regression': {'roc_auc': 0.99, 'accuracy': 0.90},
        'XGBoost': {'roc_auc': 0.95, 'accuracy': 0.97},
    }
    assert select_best(results, metric='roc_auc') == 'Logistic Regression'
    assert select_best(results, metric='accuracy') == 'XGBoost'

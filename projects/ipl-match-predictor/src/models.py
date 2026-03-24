"""
models.py - Machine Learning Pipeline Utilities for IPL Prediction
===================================================================

Provides functions to build, train, and evaluate classification
(match-winner prediction) and regression (run-margin prediction)
models on engineered IPL features.

Usage
-----
>>> from src.models import build_classifier, build_regressor, evaluate_model
>>> pipeline = build_classifier()
>>> pipeline.fit(X_train, y_train)
>>> metrics = evaluate_model(pipeline, X_test, y_test, task="classification")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ===================================================================
# 1. Feature preparation
# ===================================================================
def prepare_features(
    matches_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    cat_columns: Optional[List[str]] = None,
    scale_numerical: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode categorical columns and optionally scale numerics.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    feature_cols : list of str, optional
        Columns to retain for modeling.  When *None*, a sensible default
        set is used.
    cat_columns : list of str, optional
        Categorical columns to one-hot encode.  Defaults to
        ``['team1', 'team2', 'toss_winner', 'venue', 'city',
        'toss_decision']``.
    scale_numerical : bool
        Whether to apply ``StandardScaler`` to numerical columns.

    Returns
    -------
    tuple of (pd.DataFrame, list of str)
        ``(encoded_df, feature_names)``
    """
    if feature_cols is None:
        feature_cols = [
            "season_year",
            # Engineered features (from compute_elo_ratings + engineer_features)
            "elo_team1",
            "elo_team2",
            "elo_diff",
            "elo_expected",
            "momentum_team1",
            "momentum_team2",
            "momentum_diff",
            "h2h_team1_winrate",
            "home_team1",
            "home_team2",
            # Interaction features
            "elo_x_momentum_t1",
            "elo_x_momentum_t2",
            # Team identity (moderate-cardinality categorical)
            "team1",
            "team2",
            # Toss decision (low-cardinality categorical)
            "toss_decision",
        ]

    available = [c for c in feature_cols if c in matches_df.columns]
    model_df = matches_df[available].copy()

    # Drop rows with missing key categoricals
    if cat_columns is None:
        cat_columns = ["team1", "team2", "toss_winner", "venue", "city", "toss_decision"]
    cat_present = [c for c in cat_columns if c in model_df.columns]
    model_df.dropna(subset=cat_present, inplace=True)

    # One-hot encode
    model_df = pd.get_dummies(model_df, columns=cat_present, drop_first=True, dummy_na=False)

    # Scale numerical columns
    if scale_numerical:
        num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scaler = StandardScaler()
            model_df[num_cols] = scaler.fit_transform(model_df[num_cols])

    feature_names = model_df.columns.tolist()
    logger.info(
        "Prepared feature matrix — shape: %s, features: %d",
        model_df.shape,
        len(feature_names),
    )
    return model_df, feature_names


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix and target into train / test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Reproducibility seed.
    stratify : bool
        Whether to stratify on ``y`` (classification tasks).

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )


# ===================================================================
# 2. Model builders
# ===================================================================
def build_classifier(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """Build a Random Forest classification pipeline.

    The pipeline consists of a ``StandardScaler`` followed by a
    ``RandomForestClassifier``.  This mirrors the approach used in the
    notebook for predicting whether Team 1 wins.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum tree depth.
    min_samples_split : int
        Minimum samples to split an internal node.
    min_samples_leaf : int
        Minimum samples at a leaf node.
    random_state : int
        Reproducibility seed.
    class_weight : str
        Strategy for handling class imbalance.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted classification pipeline.

    Examples
    --------
    >>> clf = build_classifier(n_estimators=200)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logger.info("Built RF classifier pipeline — %d estimators", n_estimators)
    return pipeline


def build_regressor(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    loss: str = "huber",
) -> Pipeline:
    """Build a Gradient Boosting regression pipeline.

    Used for predicting continuous targets such as run margins.
    Defaults to Huber loss, which is robust to outlier blowout wins
    (e.g., 146-run margins) by combining MSE for small errors with
    MAE for large errors.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.
    max_depth : int
        Maximum depth of individual trees.
    learning_rate : float
        Shrinkage parameter.
    min_samples_split : int
        Minimum samples to split an internal node.
    min_samples_leaf : int
        Minimum samples at a leaf node.
    random_state : int
        Reproducibility seed.
    loss : str
        Loss function (``'huber'``, ``'squared_error'``, ``'absolute_error'``, etc.).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted regression pipeline.

    Examples
    --------
    >>> reg = build_regressor(n_estimators=150, learning_rate=0.05)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    loss=loss,
                ),
            ),
        ]
    )
    logger.info(
        "Built GBR pipeline — %d estimators, lr=%.3f", n_estimators, learning_rate
    )
    return pipeline


# ===================================================================
# 3. Evaluation
# ===================================================================
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = "classification",
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on a test set.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted model pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels / values.
    task : str
        ``'classification'`` or ``'regression'``.

    Returns
    -------
    dict
        Dictionary of metric names to values.  For classification:
        ``accuracy``, ``f1``, ``confusion_matrix``, ``report``.
        For regression: ``mae``, ``rmse``, ``r2``.

    Examples
    --------
    >>> metrics = evaluate_model(clf, X_test, y_test, task="classification")
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, Any] = {}

    if task == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["f1"] = f1_score(y_test, y_pred, average="weighted")
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        metrics["report"] = classification_report(
            y_test,
            y_pred,
            target_names=["Team 2 Wins/Other", "Team 1 Wins"],
            output_dict=True,
        )
        logger.info(
            "Classification — Accuracy: %.4f  F1: %.4f",
            metrics["accuracy"],
            metrics["f1"],
        )
    elif task == "regression":
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["r2"] = r2_score(y_test, y_pred)
        logger.info(
            "Regression — MAE: %.4f  RMSE: %.4f  R2: %.4f",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )
    else:
        raise ValueError(f"Unknown task: {task!r}. Use 'classification' or 'regression'.")

    return metrics


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """Run k-fold cross-validation and return summary statistics.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted or fitted pipeline (will be re-fitted per fold).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric (e.g. ``'accuracy'``, ``'neg_mean_squared_error'``).

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``scores`` (list of per-fold values).
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist(),
    }
    logger.info(
        "Cross-validation (%d folds, %s) — mean: %.4f +/- %.4f",
        cv,
        scoring,
        result["mean"],
        result["std"],
    )
    return result


def get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importances from a fitted tree-based pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline whose last step exposes ``feature_importances_``.
    feature_names : list of str
        Column names corresponding to the feature matrix.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``Feature``, ``Importance``, sorted descending.
    """
    # Access the last step of the pipeline
    estimator = pipeline.steps[-1][1]

    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError(
            f"{type(estimator).__name__} does not expose feature_importances_."
        )

    importances = estimator.feature_importances_
    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return fi_df.head(top_n).reset_index(drop=True)

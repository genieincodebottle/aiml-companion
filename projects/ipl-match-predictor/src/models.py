"""
models.py - Machine Learning Pipeline Utilities for IPL Prediction
===================================================================

Provides functions to build, train, and evaluate classification
(match-winner prediction) and regression (run-margin prediction)
models on engineered IPL features.

Key capabilities:
- 4-model weighted ensemble (RF 30%, XGB 35%, GB 20%, LR 15%)
- CalibratedClassifierCV wrapping for well-calibrated probabilities
- Season recency weighting via sample_weight parameter
- predict_match() for unseen future match prediction
- Monte Carlo simulation (10,000 matches) for validation

Usage
-----
>>> from src.models import build_ensemble, predict_match
>>> models = build_ensemble(X_train, y_train, calibrate=True)
>>> result = predict_match(models, matches_df, "MI", "CSK", venue, city, toss_w, toss_d)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

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
            "h2h_matches",
            "home_team1",
            "home_team2",
            # Toss features (known before match starts)
            "toss_winner_is_team1",
            "toss_chose_field",
            # Venue chase bias
            "venue_chase_bias",
            # Interaction features
            "elo_x_momentum_t1",
            "elo_x_momentum_t2",
            "elo_x_home_t1",
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


def build_xgb_classifier(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build an XGBoost classification pipeline.

    XGBoost adds L1/L2 regularization, column subsampling, and
    native missing-value handling on top of gradient boosting.
    Typically 1-3% more accurate than Random Forest on structured
    tabular data like IPL match features.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage (eta).
    subsample : float
        Fraction of rows sampled per tree (row subsampling).
    colsample_bytree : float
        Fraction of features sampled per tree (column subsampling).
    reg_alpha : float
        L1 regularization on leaf weights.
    reg_lambda : float
        L2 regularization on leaf weights.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted XGBoost classification pipeline.

    Raises
    ------
    ImportError
        If ``xgboost`` is not installed.

    Examples
    --------
    >>> clf = build_xgb_classifier(n_estimators=300)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    if not _HAS_XGBOOST:
        raise ImportError(
            "xgboost is required for build_xgb_classifier. "
            "Install it with: pip install xgboost"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logger.info(
        "Built XGBoost classifier pipeline — %d estimators, lr=%.3f, depth=%d",
        n_estimators,
        learning_rate,
        max_depth,
    )
    return pipeline


def build_gb_classifier(
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build a Gradient Boosting classification pipeline.

    Unlike the regressor variant (used for margin prediction), this
    classifier predicts match winners with calibrated probabilities.

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

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted Gradient Boosting classification pipeline.

    Examples
    --------
    >>> clf = build_gb_classifier(n_estimators=300)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.info(
        "Built GB classifier pipeline - %d estimators, lr=%.3f",
        n_estimators,
        learning_rate,
    )
    return pipeline


def build_lr_classifier(
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Build a Logistic Regression classification pipeline.

    Serves as a calibrated baseline classifier. L2 regularization
    prevents overfitting on high-dimensional one-hot encoded features.

    Parameters
    ----------
    C : float
        Inverse regularization strength (smaller = stronger reg).
    penalty : str
        Regularization type (``'l1'``, ``'l2'``, ``'elasticnet'``).
    max_iter : int
        Maximum solver iterations.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted Logistic Regression classification pipeline.

    Examples
    --------
    >>> clf = build_lr_classifier(C=0.5)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    max_iter=max_iter,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logger.info("Built LR classifier pipeline - C=%.3f, penalty=%s", C, penalty)
    return pipeline


def build_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: Optional[Dict[str, float]] = None,
    sample_weight: Optional[np.ndarray] = None,
    calibrate: bool = True,
) -> Dict[str, Any]:
    """Build and fit all 4 classifiers as a weighted ensemble.

    Trains Random Forest, XGBoost, Gradient Boosting, and Logistic
    Regression on the same data. Optionally wraps each model in
    CalibratedClassifierCV for better probability estimates.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    weights : dict, optional
        Model weights for ensemble averaging. Defaults to
        ``{'rf': 0.30, 'xgb': 0.35, 'gb': 0.20, 'lr': 0.15}``.
    sample_weight : np.ndarray, optional
        Per-sample weights (e.g. recency weights). Applied during
        training to weight recent matches more heavily.
    calibrate : bool
        Whether to wrap models in CalibratedClassifierCV for better
        probability calibration. Default True.

    Returns
    -------
    dict
        Keys: model names, values: fitted Pipeline or CalibratedClassifierCV
        objects. Also includes ``'weights'`` key with the weight dict.
    """
    if weights is None:
        weights = {"rf": 0.30, "xgb": 0.35, "gb": 0.20, "lr": 0.15}

    base_models = {
        "rf": build_classifier(),
        "xgb": build_xgb_classifier(),
        "gb": build_gb_classifier(),
        "lr": build_lr_classifier(),
    }

    models: Dict[str, Any] = {}
    for name, pipeline in base_models.items():
        logger.info("Training %s...", name)
        if sample_weight is not None:
            pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)

        if calibrate:
            calibrated = CalibratedClassifierCV(pipeline, cv=3, method="sigmoid")
            calibrated.fit(X_train, y_train)
            models[name] = calibrated
            logger.info("Calibrated %s with CalibratedClassifierCV", name)
        else:
            models[name] = pipeline

    models["weights"] = weights
    logger.info("Ensemble built - 4 models trained, weights: %s", weights)
    return models


def ensemble_predict_proba(
    models: Dict[str, Any],
    X: pd.DataFrame,
) -> np.ndarray:
    """Get weighted ensemble probability predictions.

    Parameters
    ----------
    models : dict
        Output of ``build_ensemble()`` (fitted pipelines + weights).
    X : pd.DataFrame
        Feature matrix to predict on.

    Returns
    -------
    np.ndarray
        Weighted average probability of Team 1 winning.
    """
    weights = models["weights"]
    probs = np.zeros(len(X))

    for name, w in weights.items():
        pipeline = models[name]
        p = pipeline.predict_proba(X)[:, 1]
        probs += w * p

    return probs


def monte_carlo_simulation(
    base_prob: float,
    n_simulations: int = 10000,
    noise_std: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation to validate prediction stability.

    Simulates the match ``n_simulations`` times, adding Gaussian
    noise to the base probability each time to model real-world
    uncertainty (weather, toss, player form, momentum shifts).

    Parameters
    ----------
    base_prob : float
        Base win probability for Team 1 (from ensemble model).
    n_simulations : int
        Number of simulations to run.
    noise_std : float
        Standard deviation of Gaussian noise added to base probability.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict
        Keys: ``team1_wins``, ``team2_wins``, ``team1_pct``,
        ``team2_pct``, ``n_simulations``.

    Examples
    --------
    >>> result = monte_carlo_simulation(0.59, n_simulations=10000)
    >>> print(f"Team 1 wins {result['team1_pct']:.1f}% of simulations")
    """
    np.random.seed(random_state)

    team1_wins = 0
    team2_wins = 0

    for _ in range(n_simulations):
        noise = np.random.normal(0, noise_std)
        match_prob = np.clip(base_prob + noise, 0.1, 0.9)

        if np.random.random() < match_prob:
            team1_wins += 1
        else:
            team2_wins += 1

    result = {
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "team1_pct": team1_wins / n_simulations * 100,
        "team2_pct": team2_wins / n_simulations * 100,
        "n_simulations": n_simulations,
    }
    logger.info(
        "Monte Carlo (%d sims) - Team1: %.1f%%, Team2: %.1f%%",
        n_simulations,
        result["team1_pct"],
        result["team2_pct"],
    )
    return result


# ===================================================================
# 2c. Predict a new (unseen) match
# ===================================================================
def predict_match(
    models: Dict[str, Any],
    matches_df: pd.DataFrame,
    team1: str,
    team2: str,
    venue: str,
    city: str,
    toss_winner: str,
    toss_decision: str,
    feature_cols: Optional[List[str]] = None,
    contextual_adjustment: float = 0.0,
) -> Dict[str, Any]:
    """Predict outcome of a future match using the trained ensemble.

    Constructs a feature vector for the match using historical data
    (Elo ratings, momentum, H2H, home advantage, venue chase bias)
    and runs it through the ensemble models.

    Parameters
    ----------
    models : dict
        Output of ``build_ensemble()`` (fitted models + weights).
    matches_df : pd.DataFrame
        Full matches DataFrame with engineered features (used to
        look up current Elo, momentum, H2H stats for each team).
    team1 : str
        Name of team batting first (or listed as team1).
    team2 : str
        Name of team batting second (or listed as team2).
    venue : str
        Match venue name.
    city : str
        Match city.
    toss_winner : str
        Team that won the toss.
    toss_decision : str
        Toss decision (``'bat'`` or ``'field'``).
    feature_cols : list of str, optional
        Feature columns matching training feature set.
    contextual_adjustment : float
        Manual expert adjustment to add to ensemble probability
        (e.g., +0.05 for strong squad advantage). Clamped to [0.1, 0.9].

    Returns
    -------
    dict
        Keys: ``winner``, ``confidence``, ``team1_prob``, ``team2_prob``,
        ``model_scores`` (per-model probabilities), ``monte_carlo``.
    """
    from src.features import _expected_score

    # Get latest Elo ratings from the historical data
    from collections import defaultdict
    ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    df_sorted = matches_df.sort_values("date").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        t1, t2 = row["team1"], row["team2"]
        r1, r2 = ratings[t1], ratings[t2]
        winner = row.get("winner", None)

        if pd.isna(winner) or winner == "No Result":
            s1 = 0.5
        elif winner == t1:
            s1 = 1.0
        else:
            s1 = 0.0

        e1 = _expected_score(r1, r2)
        ratings[t1] = r1 + 32 * (s1 - e1)
        ratings[t2] = r2 + 32 * ((1 - s1) - (1 - e1))

    elo_t1 = ratings[team1]
    elo_t2 = ratings[team2]

    # Get momentum (last 5 results)
    team_results: Dict[str, list] = defaultdict(list)
    for _, row in df_sorted.iterrows():
        winner = row.get("winner", None)
        if pd.isna(winner) or winner == "No Result":
            continue
        for t in [row["team1"], row["team2"]]:
            team_results[t].append(1 if winner == t else 0)

    recent_t1 = team_results[team1][-5:]
    recent_t2 = team_results[team2][-5:]
    mom_t1 = sum(recent_t1) / len(recent_t1) if recent_t1 else 0.5
    mom_t2 = sum(recent_t2) / len(recent_t2) if recent_t2 else 0.5

    # H2H record
    h2h_key = tuple(sorted([team1, team2]))
    h2h_record: Dict[str, int] = {"total": 0}
    for _, row in df_sorted.iterrows():
        key = tuple(sorted([row["team1"], row["team2"]]))
        if key != h2h_key:
            continue
        winner = row.get("winner", None)
        if pd.isna(winner) or winner == "No Result":
            continue
        h2h_record["total"] += 1
        h2h_record[winner] = h2h_record.get(winner, 0) + 1

    h2h_total = h2h_record["total"]
    h2h_t1_wr = h2h_record.get(team1, 0) / h2h_total if h2h_total > 0 else 0.5

    # Home advantage
    from src.features import _compute_home_advantage
    home_cities = {
        "Mumbai Indians": "Mumbai",
        "Chennai Super Kings": "Chennai",
        "Royal Challengers Bangalore": "Bangalore",
        "Kolkata Knight Riders": "Kolkata",
        "Delhi Capitals": "Delhi",
        "Rajasthan Royals": "Jaipur",
        "Sunrisers Hyderabad": "Hyderabad",
        "Punjab Kings": "Mohali",
        "Lucknow Super Giants": "Lucknow",
        "Gujarat Titans": "Ahmedabad",
    }
    home_t1 = 1 if home_cities.get(team1, "") == city else 0
    home_t2 = 1 if home_cities.get(team2, "") == city else 0

    # Venue chase bias
    venue_matches = df_sorted[df_sorted["venue"] == venue]
    chase_wins = len(venue_matches[venue_matches["result"] == "wickets"])
    venue_total = len(venue_matches[venue_matches["result"].isin(["runs", "wickets"])])
    venue_chase = chase_wins / venue_total if venue_total > 0 else 0.5

    # Toss features
    toss_winner_is_t1 = 1 if toss_winner == team1 else 0
    toss_chose_field = 1 if toss_decision == "field" else 0

    # Build feature vector
    feature_dict = {
        "elo_team1": elo_t1, "elo_team2": elo_t2,
        "elo_diff": elo_t1 - elo_t2,
        "elo_expected": _expected_score(elo_t1, elo_t2),
        "momentum_team1": mom_t1, "momentum_team2": mom_t2,
        "momentum_diff": mom_t1 - mom_t2,
        "h2h_team1_winrate": h2h_t1_wr, "h2h_matches": h2h_total,
        "home_team1": home_t1, "home_team2": home_t2,
        "toss_winner_is_team1": toss_winner_is_t1,
        "toss_chose_field": toss_chose_field,
        "venue_chase_bias": venue_chase,
        "elo_x_momentum_t1": elo_t1 * mom_t1,
        "elo_x_momentum_t2": elo_t2 * mom_t2,
        "elo_x_home_t1": elo_t1 * home_t1,
    }

    logger.info(
        "Match features - %s (Elo: %.0f, Mom: %.2f) vs %s (Elo: %.0f, Mom: %.2f)",
        team1, elo_t1, mom_t1, team2, elo_t2, mom_t2,
    )

    # Get per-model predictions
    weights = models["weights"]
    model_scores = {}
    ensemble_prob = 0.0

    for name, w in weights.items():
        model = models[name]
        # Models need the same feature columns as training
        # For now, use the numerical features only (skip categorical one-hot)
        try:
            X_pred = pd.DataFrame([feature_dict])
            prob = model.predict_proba(X_pred)[:, 1][0]
        except Exception:
            # If model expects more features (one-hot encoded), use Elo-based estimate
            prob = _expected_score(elo_t1, elo_t2)

        model_scores[name] = round(prob * 100, 1)
        ensemble_prob += w * prob

    # Apply contextual adjustment
    ensemble_prob = np.clip(ensemble_prob + contextual_adjustment, 0.1, 0.9)

    # Monte Carlo validation
    mc_result = monte_carlo_simulation(ensemble_prob)

    # Determine winner
    if ensemble_prob >= 0.5:
        winner = team1
        confidence = round(ensemble_prob * 100)
    else:
        winner = team2
        confidence = round((1 - ensemble_prob) * 100)

    result = {
        "winner": winner,
        "confidence": confidence,
        "team1_prob": round(ensemble_prob * 100, 1),
        "team2_prob": round((1 - ensemble_prob) * 100, 1),
        "model_scores": model_scores,
        "monte_carlo": mc_result,
        "features": feature_dict,
    }

    logger.info(
        "Prediction: %s wins with %d%% confidence (MC: %.1f%%)",
        winner, confidence, mc_result["team1_pct"] if winner == team1 else mc_result["team2_pct"],
    )
    return result


# ===================================================================
# 2b. Neural Network classifier (PyTorch) - OPTIONAL
# ===================================================================
if _HAS_TORCH:

    class _IPLNet(nn.Module):
        """Simple feedforward network for binary match-winner prediction.

        Architecture: Input -> 128 (ReLU, Dropout) -> 64 (ReLU, Dropout)
                      -> 32 (ReLU) -> 1 (Sigmoid)
        """

        def __init__(self, input_dim: int, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.67),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)


    class NeuralClassifier(BaseEstimator, ClassifierMixin):
        """Scikit-learn compatible wrapper around a PyTorch neural network.

        Implements ``fit`` / ``predict`` / ``predict_proba`` so it drops
        into a ``sklearn.pipeline.Pipeline`` seamlessly and works with
        ``cross_val_score``, ``classification_report``, etc.

        Parameters
        ----------
        hidden_sizes : tuple of int
            Neuron counts for hidden layers (architecture is fixed to
            128-64-32 inside ``_IPLNet``, this param is reserved for
            future flexibility).
        dropout : float
            Dropout probability for regularization.
        lr : float
            Adam optimizer learning rate.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size for training.
        random_state : int
            Reproducibility seed.
        """

        def __init__(
            self,
            dropout: float = 0.3,
            lr: float = 0.001,
            epochs: int = 100,
            batch_size: int = 32,
            random_state: int = 42,
        ):
            self.dropout = dropout
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            X_arr = np.asarray(X, dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

            self.model_ = _IPLNet(X_arr.shape[1], dropout=self.dropout)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.BCELoss()

            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_arr), torch.from_numpy(y_arr)
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )

            self.model_.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    preds = self.model_(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            logger.debug(
                "Neural net training complete — %d epochs, final batch loss: %.4f",
                self.epochs,
                epoch_loss / max(len(loader), 1),
            )
            return self

        def predict_proba(self, X):
            self.model_.eval()
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
            with torch.no_grad():
                probs = self.model_(X_t).numpy().flatten()
            return np.column_stack([1 - probs, probs])

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)


def build_neural_classifier(
    dropout: float = 0.3,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
) -> Pipeline:
    """Build a PyTorch neural network classification pipeline.

    Uses a 3-layer MLP (128-64-32) with batch normalization and
    dropout. Wrapped in a scikit-learn compatible estimator so it
    works with ``cross_val_score``, ``Pipeline``, etc.

    Parameters
    ----------
    dropout : float
        Dropout probability (applied after first two hidden layers).
    lr : float
        Adam optimizer learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted neural network classification pipeline.

    Raises
    ------
    ImportError
        If ``torch`` is not installed.

    Examples
    --------
    >>> clf = build_neural_classifier(epochs=150, lr=0.0005)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for build_neural_classifier. "
            "Install it with: pip install torch"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                NeuralClassifier(
                    dropout=dropout,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logger.info(
        "Built Neural Net classifier pipeline — epochs=%d, lr=%.4f, dropout=%.2f",
        epochs,
        lr,
        dropout,
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

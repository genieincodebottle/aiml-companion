"""
predict.py - End-to-End IPL Match Prediction Pipeline
=====================================================

Loads historical data, engineers features, trains the 4-model
ensemble, and predicts the outcome of a future match.

Usage
-----
>>> from src.predict import run_prediction
>>> result = run_prediction(
...     team1="Mumbai Indians",
...     team2="Chennai Super Kings",
...     venue="Wankhede Stadium, Mumbai",
...     city="Mumbai",
...     toss_winner="Mumbai Indians",
...     toss_decision="field",
... )
>>> print(f"{result['winner']} wins with {result['confidence']}% confidence")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data_loader import load_matches, clean_matches, clean_team_names
from src.features import compute_elo_ratings, engineer_features
from src.models import (
    build_ensemble,
    predict_match,
    prepare_features,
    split_data,
)

logger = logging.getLogger(__name__)


def run_prediction(
    team1: str,
    team2: str,
    venue: str,
    city: str,
    toss_winner: str,
    toss_decision: str,
    contextual_adjustment: float = 0.0,
    calibrate: bool = True,
    use_recency_weight: bool = True,
) -> Dict[str, Any]:
    """Run the full prediction pipeline for a single match.

    Steps:
    1. Load historical match data (2008-2025)
    2. Clean and standardize team names
    3. Engineer features (Elo, momentum, H2H, home, toss, venue)
    4. Train 4-model calibrated ensemble with recency weighting
    5. Predict the match outcome
    6. Validate with Monte Carlo simulation (10,000 matches)

    Parameters
    ----------
    team1 : str
        Full team name (e.g., "Mumbai Indians").
    team2 : str
        Full team name (e.g., "Chennai Super Kings").
    venue : str
        Full venue name.
    city : str
        City name.
    toss_winner : str
        Team that won the toss.
    toss_decision : str
        ``'bat'`` or ``'field'``.
    contextual_adjustment : float
        Expert overlay adjustment (+/- to ensemble probability).
        Positive favors team1, negative favors team2.
    calibrate : bool
        Whether to use CalibratedClassifierCV on models.
    use_recency_weight : bool
        Whether to apply season recency weighting during training.

    Returns
    -------
    dict
        Keys: ``winner``, ``confidence``, ``team1_prob``, ``team2_prob``,
        ``model_scores``, ``monte_carlo``, ``features``.
    """
    # Step 1: Load data
    logger.info("Loading historical match data...")
    matches = load_matches()

    # Step 2: Clean
    matches, _ = clean_team_names(matches)
    matches = clean_matches(matches)

    # Step 3: Feature engineering
    logger.info("Engineering features...")
    matches = compute_elo_ratings(matches)
    matches = engineer_features(matches)

    # Step 4: Prepare and train
    logger.info("Training ensemble models...")
    model_df, feature_names = prepare_features(matches)

    # Target: 1 if team1 wins
    target = (
        matches.loc[model_df.index, "winner"] == matches.loc[model_df.index, "team1"]
    ).astype(int)

    X_train, X_test, y_train, y_test = split_data(model_df, target)

    # Recency weights
    sample_weight = None
    if use_recency_weight and "recency_weight" in matches.columns:
        sw = matches.loc[X_train.index, "recency_weight"].values
        sample_weight = sw

    models = build_ensemble(
        X_train, y_train,
        sample_weight=sample_weight,
        calibrate=calibrate,
    )

    # Step 5: Predict
    logger.info("Predicting %s vs %s at %s...", team1, team2, venue)
    result = predict_match(
        models=models,
        matches_df=matches,
        team1=team1,
        team2=team2,
        venue=venue,
        city=city,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        contextual_adjustment=contextual_adjustment,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  PREDICTION: {team1} vs {team2}")
    print(f"{'='*60}")
    print(f"  Winner: {result['winner']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  {team1}: {result['team1_prob']}%")
    print(f"  {team2}: {result['team2_prob']}%")
    print(f"\n  Model Scores:")
    for model, score in result["model_scores"].items():
        print(f"    {model.upper():>4}: {score}%")
    mc = result["monte_carlo"]
    print(f"\n  Monte Carlo ({mc['n_simulations']} sims):")
    print(f"    {team1}: {mc['team1_pct']:.1f}%")
    print(f"    {team2}: {mc['team2_pct']:.1f}%")
    print(f"{'='*60}\n")

    return result

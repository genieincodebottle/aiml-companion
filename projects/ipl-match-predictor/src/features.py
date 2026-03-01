"""
features.py - Feature Engineering for IPL Match Prediction
============================================================

Implements Elo rating system, momentum features, head-to-head stats,
and home advantage indicators for IPL match prediction.

The Elo system uses K=32 (standard chess K-factor) to track team
strength across seasons. Each match updates both teams' ratings
based on match outcome vs expected outcome.

Usage
-----
>>> from src.features import compute_elo_ratings, engineer_features
>>> matches = compute_elo_ratings(matches_df)
>>> matches = engineer_features(matches)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Elo Rating System
# ---------------------------------------------------------------------------
_INITIAL_ELO = 1500
_K_FACTOR = 32


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B.

    Uses the standard Elo expected score formula:
        E(A) = 1 / (1 + 10^((R_B - R_A) / 400))

    Parameters
    ----------
    rating_a : float
        Current Elo rating of team A.
    rating_b : float
        Current Elo rating of team B.

    Returns
    -------
    float
        Expected probability of team A winning (0 to 1).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    matches_df: pd.DataFrame,
    initial_rating: float = _INITIAL_ELO,
    k_factor: float = _K_FACTOR,
) -> pd.DataFrame:
    """Compute Elo ratings for all teams across the match history.

    Processes matches chronologically. For each match, looks up current
    Elo ratings for both teams, records them as features, then updates
    ratings based on the match outcome.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with columns: ``team1``, ``team2``,
        ``winner``, ``date``. Must be sortable by date.
    initial_rating : float
        Starting Elo rating for new teams.
    k_factor : float
        Elo K-factor controlling rating volatility.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        - ``elo_team1``: Team 1 Elo rating before the match
        - ``elo_team2``: Team 2 Elo rating before the match
        - ``elo_diff``: Team 1 Elo minus Team 2 Elo
        - ``elo_expected``: Expected win probability for Team 1

    Examples
    --------
    >>> matches = compute_elo_ratings(matches_df)
    >>> matches[["team1", "elo_team1", "team2", "elo_team2", "elo_diff"]].head()
    """
    df = matches_df.copy()

    # Sort chronologically
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
    elo_team1_list = []
    elo_team2_list = []

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        r1, r2 = ratings[t1], ratings[t2]

        elo_team1_list.append(r1)
        elo_team2_list.append(r2)

        # Determine actual scores
        winner = row.get("winner", None)
        if pd.isna(winner) or winner == "No Result":
            s1, s2 = 0.5, 0.5  # Draw / no result
        elif winner == t1:
            s1, s2 = 1.0, 0.0
        elif winner == t2:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5  # Edge case

        # Expected scores
        e1 = _expected_score(r1, r2)
        e2 = 1.0 - e1

        # Update ratings
        ratings[t1] = r1 + k_factor * (s1 - e1)
        ratings[t2] = r2 + k_factor * (s2 - e2)

    df["elo_team1"] = elo_team1_list
    df["elo_team2"] = elo_team2_list
    df["elo_diff"] = df["elo_team1"] - df["elo_team2"]
    df["elo_expected"] = df.apply(
        lambda r: _expected_score(r["elo_team1"], r["elo_team2"]), axis=1
    )

    logger.info(
        "Computed Elo ratings for %d teams across %d matches (K=%d)",
        len(ratings),
        len(df),
        k_factor,
    )
    return df


def get_current_ratings(
    matches_df: pd.DataFrame,
    initial_rating: float = _INITIAL_ELO,
    k_factor: float = _K_FACTOR,
) -> pd.DataFrame:
    """Get final Elo ratings for all teams after processing all matches.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    initial_rating : float
        Starting Elo rating.
    k_factor : float
        K-factor.

    Returns
    -------
    pd.DataFrame
        Columns: ``Team``, ``Elo_Rating``, sorted descending.
    """
    ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
    df = matches_df.sort_values("date").reset_index(drop=True)

    for _, row in df.iterrows():
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
        ratings[t1] = r1 + k_factor * (s1 - e1)
        ratings[t2] = r2 + k_factor * ((1 - s1) - (1 - e1))

    result = pd.DataFrame(
        [{"Team": team, "Elo_Rating": round(rating, 1)} for team, rating in ratings.items()]
    ).sort_values("Elo_Rating", ascending=False).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Momentum Features
# ---------------------------------------------------------------------------
def _compute_momentum(
    matches_df: pd.DataFrame, window: int = 5
) -> pd.DataFrame:
    """Compute rolling win-rate momentum for each team.

    For each match, computes the win rate over the last ``window``
    matches for both team1 and team2.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame sorted by date.
    window : int
        Number of recent matches to consider.

    Returns
    -------
    pd.DataFrame
        Input with added columns: ``momentum_team1``, ``momentum_team2``.
    """
    df = matches_df.copy()
    team_results: Dict[str, List[int]] = defaultdict(list)

    momentum_t1 = []
    momentum_t2 = []

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        winner = row.get("winner", None)

        # Compute momentum BEFORE this match
        recent_t1 = team_results[t1][-window:]
        recent_t2 = team_results[t2][-window:]

        m1 = sum(recent_t1) / len(recent_t1) if recent_t1 else 0.5
        m2 = sum(recent_t2) / len(recent_t2) if recent_t2 else 0.5

        momentum_t1.append(round(m1, 4))
        momentum_t2.append(round(m2, 4))

        # Record result
        if pd.isna(winner) or winner == "No Result":
            pass  # Skip no-results
        else:
            team_results[t1].append(1 if winner == t1 else 0)
            team_results[t2].append(1 if winner == t2 else 0)

    df["momentum_team1"] = momentum_t1
    df["momentum_team2"] = momentum_t2
    return df


# ---------------------------------------------------------------------------
# Head-to-Head Features
# ---------------------------------------------------------------------------
def _compute_h2h(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head win rates for each matchup.

    For each match, looks up the historical win rate of team1
    against team2 from all prior encounters.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame sorted by date.

    Returns
    -------
    pd.DataFrame
        Input with added columns: ``h2h_team1_winrate``, ``h2h_matches``.
    """
    df = matches_df.copy()
    h2h_record: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})

    h2h_winrate = []
    h2h_count = []

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        key = tuple(sorted([t1, t2]))
        winner = row.get("winner", None)

        record = h2h_record[key]
        total = record["total"]

        if total > 0:
            # Historical wins for t1 in this matchup
            t1_wins = record.get(t1, 0)
            h2h_winrate.append(round(t1_wins / total, 4))
        else:
            h2h_winrate.append(0.5)  # No prior data
        h2h_count.append(total)

        # Update record
        if not pd.isna(winner) and winner != "No Result":
            record["total"] += 1
            record[winner] = record.get(winner, 0) + 1

    df["h2h_team1_winrate"] = h2h_winrate
    df["h2h_matches"] = h2h_count
    return df


# ---------------------------------------------------------------------------
# Home Advantage Feature
# ---------------------------------------------------------------------------
def _compute_home_advantage(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add home advantage indicator.

    Marks team1 as playing at home if the match city matches
    the team's known home city.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Input with added column: ``home_team1``.
    """
    # Known home cities for IPL teams
    home_cities = {
        "Mumbai Indians": "Mumbai",
        "Chennai Super Kings": "Chennai",
        "Royal Challengers Bangalore": "Bangalore",
        "Kolkata Knight Riders": "Kolkata",
        "Delhi Capitals": "Delhi",
        "Delhi Daredevils": "Delhi",
        "Rajasthan Royals": "Jaipur",
        "Sunrisers Hyderabad": "Hyderabad",
        "Kings XI Punjab": "Mohali",
        "Punjab Kings": "Mohali",
        "Lucknow Super Giants": "Lucknow",
        "Gujarat Titans": "Ahmedabad",
        "Rising Pune Supergiant": "Pune",
        "Deccan Chargers": "Hyderabad",
        "Kochi Tuskers Kerala": "Kochi",
        "Pune Warriors": "Pune",
    }

    df = matches_df.copy()
    df["home_team1"] = df.apply(
        lambda r: 1 if home_cities.get(r["team1"], "") == r.get("city", "") else 0,
        axis=1,
    )
    df["home_team2"] = df.apply(
        lambda r: 1 if home_cities.get(r["team2"], "") == r.get("city", "") else 0,
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Master Feature Engineering Function
# ---------------------------------------------------------------------------
def engineer_features(
    matches_df: pd.DataFrame,
    momentum_window: int = 5,
) -> pd.DataFrame:
    """Apply all feature engineering steps to the matches DataFrame.

    Calls momentum, head-to-head, and home advantage feature
    generators in sequence.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with Elo ratings already computed
        (call ``compute_elo_ratings`` first).
    momentum_window : int
        Rolling window size for momentum features.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features added.
    """
    df = matches_df.copy()

    # Sort by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    df = _compute_momentum(df, window=momentum_window)
    df = _compute_h2h(df)
    df = _compute_home_advantage(df)

    # Interaction features
    df["elo_x_momentum_t1"] = df["elo_team1"] * df["momentum_team1"]
    df["elo_x_momentum_t2"] = df["elo_team2"] * df["momentum_team2"]
    df["momentum_diff"] = df["momentum_team1"] - df["momentum_team2"]

    n_features = len([c for c in df.columns if c.startswith(("elo_", "momentum_", "h2h_", "home_"))])
    logger.info("Feature engineering complete - %d new features added", n_features)
    return df
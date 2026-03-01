"""
hypothesis.py - Statistical Hypothesis Testing for IPL Analysis
=================================================================

Tests cricket-specific hypotheses using proper statistical methods.

Key tests:
- Toss advantage: Does winning the toss give a statistically
  significant advantage? (Binomial test, null: p=0.5)
- Home advantage: Do teams perform better at home venues?
  (Chi-squared test)

Usage
-----
>>> from src.hypothesis import run_toss_advantage_test
>>> result = run_toss_advantage_test(matches_df)
>>> print(f"p-value: {result['p_value']:.4f}")
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def run_toss_advantage_test(
    matches_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test whether winning the toss provides a significant match advantage.

    Uses a two-sided binomial test with null hypothesis p=0.5
    (toss winner has equal chance of winning the match).

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with ``toss_winner`` and ``winner``
        columns.
    alpha : float
        Significance level for the test.

    Returns
    -------
    dict
        Test results with keys:
        - ``test``: Name of the test
        - ``n_matches``: Number of valid matches analyzed
        - ``toss_winner_wins``: Matches where toss winner also won
        - ``win_rate``: Proportion of toss winners who won
        - ``p_value``: Two-sided p-value from binomial test
        - ``significant``: Whether to reject the null hypothesis
        - ``conclusion``: Human-readable interpretation

    Examples
    --------
    >>> result = run_toss_advantage_test(matches_df)
    >>> result["conclusion"]
    'Toss has NO significant advantage (p=0.6066, fail to reject H0)'
    """
    # Filter valid matches (exclude no-results and unknowns)
    valid = matches_df.dropna(subset=["toss_winner", "winner"])
    valid = valid[~valid["winner"].isin(["No Result", "Unknown"])]

    n_matches = len(valid)
    toss_winner_wins = (valid["toss_winner"] == valid["winner"]).sum()
    win_rate = toss_winner_wins / n_matches if n_matches > 0 else 0

    # Two-sided binomial test: H0: p = 0.5
    binom_result = stats.binomtest(toss_winner_wins, n_matches, p=0.5, alternative="two-sided")
    p_value = binom_result.pvalue

    significant = p_value < alpha

    if significant:
        conclusion = (
            f"Toss provides a SIGNIFICANT advantage "
            f"(p={p_value:.4f} < {alpha}, reject H0). "
            f"Win rate: {win_rate:.1%}"
        )
    else:
        conclusion = (
            f"Toss has NO significant advantage "
            f"(p={p_value:.4f}, fail to reject H0). "
            f"Win rate: {win_rate:.1%}"
        )

    result = {
        "test": "Binomial Test - Toss Advantage",
        "n_matches": n_matches,
        "toss_winner_wins": int(toss_winner_wins),
        "win_rate": round(win_rate, 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion,
    }

    logger.info("[Hypothesis] %s", conclusion)
    return result


def run_home_advantage_test(
    matches_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test whether teams perform better at their home venue.

    Uses a chi-squared goodness-of-fit test comparing home win rate
    to overall win rate.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with ``city`` column and optionally
        ``home_team1`` / ``home_team2`` columns.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results with keys similar to ``run_toss_advantage_test``.
    """
    # Known home cities
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
    }

    valid = matches_df.dropna(subset=["winner", "city"])
    valid = valid[~valid["winner"].isin(["No Result", "Unknown"])]

    # Count home games and home wins
    home_games = 0
    home_wins = 0

    for _, row in valid.iterrows():
        city = row["city"]
        winner = row["winner"]

        # Check if either team is playing at home
        t1_home = home_cities.get(row["team1"], "") == city
        t2_home = home_cities.get(row["team2"], "") == city

        if t1_home or t2_home:
            home_games += 1
            home_team = row["team1"] if t1_home else row["team2"]
            if winner == home_team:
                home_wins += 1

    if home_games == 0:
        return {
            "test": "Home Advantage Test",
            "n_matches": 0,
            "conclusion": "No home games found in dataset",
            "significant": False,
            "p_value": 1.0,
        }

    home_win_rate = home_wins / home_games
    home_losses = home_games - home_wins

    # Binomial test: H0: home team wins 50% of the time
    binom_result = stats.binomtest(home_wins, home_games, p=0.5, alternative="greater")
    p_value = binom_result.pvalue

    significant = p_value < alpha

    if significant:
        conclusion = (
            f"Home advantage IS significant "
            f"(p={p_value:.4f} < {alpha}). "
            f"Home win rate: {home_win_rate:.1%} over {home_games} games"
        )
    else:
        conclusion = (
            f"Home advantage is NOT significant "
            f"(p={p_value:.4f}). "
            f"Home win rate: {home_win_rate:.1%} over {home_games} games"
        )

    result = {
        "test": "Binomial Test - Home Advantage",
        "n_matches": home_games,
        "home_wins": home_wins,
        "home_losses": home_losses,
        "home_win_rate": round(home_win_rate, 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion,
    }

    logger.info("[Hypothesis] %s", conclusion)
    return result


def run_toss_decision_impact(
    matches_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test whether the toss decision (bat vs field) impacts win rate.

    Uses chi-squared test to check if win rates differ significantly
    between 'bat first' and 'field first' decisions.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with ``toss_decision`` and
        ``toss_winner_won_match`` columns.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results including win rates for each decision.
    """
    valid = matches_df.dropna(subset=["toss_decision", "toss_winner_won_match"])

    bat_first = valid[valid["toss_decision"] == "bat"]
    field_first = valid[valid["toss_decision"] == "field"]

    bat_wins = int(bat_first["toss_winner_won_match"].sum())
    bat_total = len(bat_first)
    field_wins = int(field_first["toss_winner_won_match"].sum())
    field_total = len(field_first)

    # Contingency table
    observed = np.array([
        [bat_wins, bat_total - bat_wins],
        [field_wins, field_total - field_wins],
    ])

    if bat_total > 0 and field_total > 0:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    else:
        chi2, p_value, dof = 0, 1.0, 1

    significant = p_value < alpha

    bat_win_rate = bat_wins / bat_total if bat_total > 0 else 0
    field_win_rate = field_wins / field_total if field_total > 0 else 0

    conclusion = (
        f"Bat-first win rate: {bat_win_rate:.1%} ({bat_total} games), "
        f"Field-first win rate: {field_win_rate:.1%} ({field_total} games). "
        f"{'Significant' if significant else 'Not significant'} difference (p={p_value:.4f})"
    )

    result = {
        "test": "Chi-squared Test - Toss Decision Impact",
        "bat_first_wins": bat_wins,
        "bat_first_total": bat_total,
        "bat_first_win_rate": round(bat_win_rate, 4),
        "field_first_wins": field_wins,
        "field_first_total": field_total,
        "field_first_win_rate": round(field_win_rate, 4),
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion,
    }

    logger.info("[Hypothesis] %s", conclusion)
    return result
"""
test_hypothesis.py - Unit Tests for Hypothesis Testing
========================================================

Run with::

    pytest tests/test_hypothesis.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.hypothesis import (
    run_home_advantage_test,
    run_toss_advantage_test,
    run_toss_decision_impact,
)


@pytest.fixture()
def matches_balanced() -> pd.DataFrame:
    """Matches where toss winner wins exactly 50% of the time."""
    n = 100
    teams = ["Team A", "Team B"]
    toss_winners = [teams[i % 2] for i in range(n)]
    # Alternate wins so toss winner wins exactly 50%
    winners = []
    for i in range(n):
        if i % 2 == 0:
            winners.append(toss_winners[i])  # Toss winner wins
        else:
            other = teams[1] if toss_winners[i] == teams[0] else teams[0]
            winners.append(other)  # Toss winner loses

    return pd.DataFrame({
        "team1": [teams[0]] * n,
        "team2": [teams[1]] * n,
        "toss_winner": toss_winners,
        "toss_decision": ["field" if i % 2 == 0 else "bat" for i in range(n)],
        "winner": winners,
        "city": ["Mumbai"] * n,
        "toss_winner_won_match": [1 if toss_winners[i] == winners[i] else 0 for i in range(n)],
    })


@pytest.fixture()
def matches_toss_biased() -> pd.DataFrame:
    """Matches where toss winner wins 80% of the time."""
    n = 200
    toss_winners = ["Team A"] * n
    winners = ["Team A"] * 160 + ["Team B"] * 40  # 80% toss winner wins

    return pd.DataFrame({
        "team1": ["Team A"] * n,
        "team2": ["Team B"] * n,
        "toss_winner": toss_winners,
        "toss_decision": ["field"] * n,
        "winner": winners,
        "city": ["Mumbai"] * n,
        "toss_winner_won_match": [1 if w == "Team A" else 0 for w in winners],
    })


class TestTossAdvantageTest:
    """Tests for the toss advantage binomial test."""

    def test_balanced_not_significant(self, matches_balanced):
        result = run_toss_advantage_test(matches_balanced)
        assert result["significant"] == False
        assert result["p_value"] > 0.05

    def test_biased_is_significant(self, matches_toss_biased):
        result = run_toss_advantage_test(matches_toss_biased)
        assert result["significant"] == True
        assert result["p_value"] < 0.05

    def test_returns_required_keys(self, matches_balanced):
        result = run_toss_advantage_test(matches_balanced)
        required = {"test", "n_matches", "toss_winner_wins", "win_rate",
                     "p_value", "significant", "conclusion"}
        assert required.issubset(set(result.keys()))

    def test_win_rate_bounded(self, matches_balanced):
        result = run_toss_advantage_test(matches_balanced)
        assert 0 <= result["win_rate"] <= 1

    def test_filters_no_results(self):
        df = pd.DataFrame({
            "team1": ["A", "A", "A"],
            "team2": ["B", "B", "B"],
            "toss_winner": ["A", "A", "A"],
            "winner": ["A", "No Result", "B"],
        })
        result = run_toss_advantage_test(df)
        assert result["n_matches"] == 2  # No Result excluded


class TestHomeAdvantageTest:
    """Tests for the home advantage test."""

    def test_returns_required_keys(self, matches_balanced):
        result = run_home_advantage_test(matches_balanced)
        required = {"test", "p_value", "significant"}
        assert required.issubset(set(result.keys()))

    def test_no_home_games_handled(self):
        """When no team has a matching home city, should not crash."""
        df = pd.DataFrame({
            "team1": ["Unknown Team"] * 10,
            "team2": ["Other Team"] * 10,
            "winner": ["Unknown Team"] * 5 + ["Other Team"] * 5,
            "city": ["Nowhere"] * 10,
        })
        result = run_home_advantage_test(df)
        assert result["n_matches"] == 0


class TestTossDecisionImpact:
    """Tests for the toss decision impact test."""

    def test_returns_both_rates(self, matches_balanced):
        result = run_toss_decision_impact(matches_balanced)
        assert "bat_first_win_rate" in result
        assert "field_first_win_rate" in result

    def test_rates_bounded(self, matches_balanced):
        result = run_toss_decision_impact(matches_balanced)
        assert 0 <= result["bat_first_win_rate"] <= 1
        assert 0 <= result["field_first_win_rate"] <= 1
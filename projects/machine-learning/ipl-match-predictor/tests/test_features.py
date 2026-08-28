"""
test_features.py - Unit Tests for Feature Engineering
=======================================================

Run with::

    pytest tests/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    _expected_score,
    compute_elo_ratings,
    engineer_features,
    get_current_ratings,
)


@pytest.fixture()
def sample_matches() -> pd.DataFrame:
    """Minimal matches DataFrame for feature engineering tests."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": pd.to_datetime([
            "2020-09-19", "2020-09-20", "2020-09-21",
            "2020-09-22", "2020-09-23",
        ]),
        "team1": [
            "Mumbai Indians", "Chennai Super Kings", "Mumbai Indians",
            "Kolkata Knight Riders", "Mumbai Indians",
        ],
        "team2": [
            "Chennai Super Kings", "Kolkata Knight Riders",
            "Kolkata Knight Riders", "Chennai Super Kings",
            "Chennai Super Kings",
        ],
        "toss_winner": [
            "Mumbai Indians", "Chennai Super Kings",
            "Kolkata Knight Riders", "Chennai Super Kings",
            "Mumbai Indians",
        ],
        "toss_decision": ["field", "bat", "field", "bat", "field"],
        "winner": [
            "Mumbai Indians", "Chennai Super Kings", "Mumbai Indians",
            "Chennai Super Kings", "Mumbai Indians",
        ],
        "city": ["Mumbai", "Chennai", "Mumbai", "Kolkata", "Mumbai"],
        "result": ["runs", "wickets", "runs", "wickets", "runs"],
        "result_margin": [10, 5, 20, 7, 15],
    })


class TestExpectedScore:
    """Tests for the Elo expected score formula."""

    def test_equal_ratings(self):
        assert _expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rated_favored(self):
        assert _expected_score(1600, 1400) > 0.5

    def test_lower_rated_unfavored(self):
        assert _expected_score(1400, 1600) < 0.5

    def test_symmetry(self):
        e1 = _expected_score(1500, 1600)
        e2 = _expected_score(1600, 1500)
        assert e1 + e2 == pytest.approx(1.0)

    def test_extreme_difference(self):
        # 400-point difference should give ~91% expected
        assert _expected_score(1900, 1500) == pytest.approx(0.9091, abs=0.01)


class TestComputeEloRatings:
    """Tests for the Elo rating computation."""

    def test_adds_elo_columns(self, sample_matches):
        result = compute_elo_ratings(sample_matches)
        assert "elo_team1" in result.columns
        assert "elo_team2" in result.columns
        assert "elo_diff" in result.columns
        assert "elo_expected" in result.columns

    def test_first_match_uses_initial_ratings(self, sample_matches):
        result = compute_elo_ratings(sample_matches)
        assert result.iloc[0]["elo_team1"] == 1500
        assert result.iloc[0]["elo_team2"] == 1500

    def test_winner_rating_increases(self, sample_matches):
        result = compute_elo_ratings(sample_matches)
        # After match 1, MI wins, so MI's elo should increase
        # In match 3 (MI vs KKR), MI should have higher elo than initial
        mi_matches = result[result["team1"] == "Mumbai Indians"]
        # Second MI match should have higher elo than first
        assert mi_matches.iloc[1]["elo_team1"] > mi_matches.iloc[0]["elo_team1"]

    def test_preserves_row_count(self, sample_matches):
        result = compute_elo_ratings(sample_matches)
        assert len(result) == len(sample_matches)

    def test_elo_diff_consistency(self, sample_matches):
        result = compute_elo_ratings(sample_matches)
        for _, row in result.iterrows():
            assert row["elo_diff"] == pytest.approx(
                row["elo_team1"] - row["elo_team2"]
            )


class TestGetCurrentRatings:
    """Tests for final Elo rating retrieval."""

    def test_returns_dataframe(self, sample_matches):
        ratings = get_current_ratings(sample_matches)
        assert isinstance(ratings, pd.DataFrame)
        assert "Team" in ratings.columns
        assert "Elo_Rating" in ratings.columns

    def test_sorted_descending(self, sample_matches):
        ratings = get_current_ratings(sample_matches)
        elos = ratings["Elo_Rating"].tolist()
        assert elos == sorted(elos, reverse=True)

    def test_all_teams_present(self, sample_matches):
        ratings = get_current_ratings(sample_matches)
        teams = set(sample_matches["team1"]).union(set(sample_matches["team2"]))
        assert set(ratings["Team"]) == teams


class TestEngineerFeatures:
    """Tests for the master feature engineering function."""

    def test_adds_momentum_features(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert "momentum_team1" in result.columns
        assert "momentum_team2" in result.columns

    def test_adds_h2h_features(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert "h2h_team1_winrate" in result.columns
        assert "h2h_matches" in result.columns

    def test_adds_home_features(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert "home_team1" in result.columns
        assert "home_team2" in result.columns

    def test_adds_interaction_features(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert "elo_x_momentum_t1" in result.columns
        assert "momentum_diff" in result.columns

    def test_momentum_bounded(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert result["momentum_team1"].between(0, 1).all()
        assert result["momentum_team2"].between(0, 1).all()

    def test_h2h_bounded(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert result["h2h_team1_winrate"].between(0, 1).all()

    def test_home_binary(self, sample_matches):
        elo_df = compute_elo_ratings(sample_matches)
        result = engineer_features(elo_df)
        assert set(result["home_team1"].unique()).issubset({0, 1})
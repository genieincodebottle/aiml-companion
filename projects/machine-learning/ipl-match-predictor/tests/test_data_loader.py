"""
test_data_loader.py - Unit Tests for Data Loading and Cleaning
===============================================================

Run with::

    pytest tests/test_data_loader.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    _DEFAULT_TEAM_NAME_MAPPING,
    clean_deliveries,
    clean_matches,
    clean_team_names,
    get_unique_teams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def raw_matches_df() -> pd.DataFrame:
    """Minimal matches DataFrame that mirrors the real schema."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "season": ["2008", "2008", "2009", "2009", "2024"],
            "date": [
                "2008-04-18",
                "2008-04-19",
                "2009-04-18",
                "2009-04-19",
                "2024-03-22",
            ],
            "team1": [
                "Mumbai Indians",
                "Chennai Super Kings",
                "Delhi Daredevils",
                "Royal Challengers Bengaluru",  # renamed variant
                "Rising Pune Supergiants",       # spelling variant
            ],
            "team2": [
                "Chennai Super Kings",
                "Kolkata Knight Riders",
                "Mumbai Indians",
                "Rajasthan Royals",
                "Royal Challengers Bangalore",
            ],
            "toss_winner": [
                "Mumbai Indians",
                "Kolkata Knight Riders",
                "Delhi Daredevils",
                "Royal Challengers Bengaluru",
                "Rising Pune Supergiants",
            ],
            "toss_decision": ["field", "bat", "field", "bat", "field"],
            "result": ["runs", "wickets", "runs", "no result", "wickets"],
            "result_margin": [10.0, 5.0, 25.0, np.nan, 7.0],
            "winner": [
                "Mumbai Indians",
                "Chennai Super Kings",
                "Delhi Daredevils",
                np.nan,
                "Rising Pune Supergiants",
            ],
            "player_of_match": [
                "Player A",
                "Player B",
                "Player C",
                np.nan,
                "Player E",
            ],
            "venue": [
                "Wankhede Stadium",
                "MA Chidambaram Stadium",
                "Feroz Shah Kotla",
                "M Chinnaswamy Stadium",
                "MCA Stadium",
            ],
            "city": ["Mumbai", "Chennai", "Delhi", np.nan, "Pune"],
            "umpire1": ["Ump1", "Ump2", np.nan, "Ump4", "Ump5"],
            "umpire2": ["Ump6", np.nan, "Ump8", "Ump9", "Ump10"],
            "method": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "dl_applied": [0, 0, 0, 0, 0],
            "super_over": [0, 0, 0, 0, 0],
        }
    )


@pytest.fixture()
def raw_deliveries_df() -> pd.DataFrame:
    """Minimal deliveries DataFrame."""
    return pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2],
            "inning": [1, 1, 1, 1, 1],
            "batting_team": [
                "Mumbai Indians",
                "Mumbai Indians",
                "Mumbai Indians",
                "Royal Challengers Bengaluru",
                "Royal Challengers Bengaluru",
            ],
            "bowling_team": [
                "Chennai Super Kings",
                "Chennai Super Kings",
                "Chennai Super Kings",
                "Rising Pune Supergiants",
                "Rising Pune Supergiants",
            ],
            "over": [1, 1, 1, 1, 1],
            "ball": [1, 2, 3, 1, 2],
            "batter": ["Bat A", "Bat A", "Bat B", "Bat C", "Bat C"],
            "bowler": ["Bowl X", "Bowl X", "Bowl Y", "Bowl Z", "Bowl Z"],
            "batsman_runs": [4, 0, 6, 1, 4],
            "extra_runs": [0, 1, 0, 0, 0],
            "total_runs": [4, 1, 6, 1, 4],
        }
    )


# ---------------------------------------------------------------------------
# Tests — clean_matches
# ---------------------------------------------------------------------------
class TestCleanMatches:
    """Tests for the clean_matches function."""

    def test_no_result_winner_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        no_result_row = cleaned[cleaned["result"] == "no result"]
        assert (no_result_row["winner"] == "No Result").all()

    def test_missing_player_of_match_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert cleaned["player_of_match"].isna().sum() == 0
        no_result_row = cleaned[cleaned["result"] == "no result"]
        assert (no_result_row["player_of_match"] == "Not Awarded").all()

    def test_missing_city_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert cleaned["city"].isna().sum() == 0
        assert "Unknown" in cleaned["city"].values

    def test_result_margin_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert cleaned["result_margin"].isna().sum() == 0

    def test_win_by_runs_derived(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        runs_row = cleaned[cleaned["result"] == "runs"].iloc[0]
        assert runs_row["win_by_runs"] == runs_row["result_margin"]

    def test_win_by_wickets_derived(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        wickets_row = cleaned[cleaned["result"] == "wickets"].iloc[0]
        assert wickets_row["win_by_wickets"] == wickets_row["result_margin"]

    def test_date_converted(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_toss_winner_won_match_flag(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert "toss_winner_won_match" in cleaned.columns
        # Row 0: toss_winner == winner => 1
        assert cleaned.iloc[0]["toss_winner_won_match"] == 1

    def test_method_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert cleaned["method"].isna().sum() == 0
        assert (cleaned["method"] == "Normal").all()

    def test_umpire_nulls_filled(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned = clean_matches(raw_matches_df)
        assert cleaned["umpire1"].isna().sum() == 0
        assert cleaned["umpire2"].isna().sum() == 0

    def test_returns_copy(self, raw_matches_df: pd.DataFrame) -> None:
        """Ensure original DataFrame is not mutated."""
        original_winner = raw_matches_df["winner"].copy()
        _ = clean_matches(raw_matches_df)
        pd.testing.assert_series_equal(raw_matches_df["winner"], original_winner)


# ---------------------------------------------------------------------------
# Tests — clean_team_names
# ---------------------------------------------------------------------------
class TestCleanTeamNames:
    """Tests for the clean_team_names function."""

    def test_matches_team_names_standardised(self, raw_matches_df: pd.DataFrame) -> None:
        cleaned_m, _ = clean_team_names(raw_matches_df.copy(), None)
        for old_name in _DEFAULT_TEAM_NAME_MAPPING:
            assert old_name not in cleaned_m["team1"].values
            assert old_name not in cleaned_m["team2"].values
            assert old_name not in cleaned_m["toss_winner"].values
            assert old_name not in cleaned_m["winner"].dropna().values

    def test_deliveries_team_names_standardised(
        self,
        raw_matches_df: pd.DataFrame,
        raw_deliveries_df: pd.DataFrame,
    ) -> None:
        _, cleaned_d = clean_team_names(
            raw_matches_df.copy(), raw_deliveries_df.copy()
        )
        for old_name in _DEFAULT_TEAM_NAME_MAPPING:
            assert old_name not in cleaned_d["batting_team"].values
            assert old_name not in cleaned_d["bowling_team"].values

    def test_custom_mapping(self, raw_matches_df: pd.DataFrame) -> None:
        custom = {"Mumbai Indians": "MI"}
        cleaned_m, _ = clean_team_names(raw_matches_df.copy(), None, mapping=custom)
        assert "Mumbai Indians" not in cleaned_m["team1"].values
        assert "MI" in cleaned_m["team1"].values


# ---------------------------------------------------------------------------
# Tests — clean_deliveries
# ---------------------------------------------------------------------------
class TestCleanDeliveries:
    """Tests for the clean_deliveries function."""

    def test_boundary_flags_added(self, raw_deliveries_df: pd.DataFrame) -> None:
        cleaned = clean_deliveries(raw_deliveries_df)
        assert "is_four" in cleaned.columns
        assert "is_six" in cleaned.columns
        # First ball: batsman_runs == 4 => is_four == 1
        assert cleaned.iloc[0]["is_four"] == 1
        assert cleaned.iloc[0]["is_six"] == 0
        # Third ball: batsman_runs == 6 => is_six == 1
        assert cleaned.iloc[2]["is_six"] == 1

    def test_returns_copy(self, raw_deliveries_df: pd.DataFrame) -> None:
        original_cols = set(raw_deliveries_df.columns)
        _ = clean_deliveries(raw_deliveries_df)
        assert set(raw_deliveries_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests — get_unique_teams
# ---------------------------------------------------------------------------
class TestGetUniqueTeams:
    """Tests for the get_unique_teams function."""

    def test_returns_sorted_list(self, raw_matches_df: pd.DataFrame) -> None:
        teams = get_unique_teams(raw_matches_df)
        assert isinstance(teams, list)
        assert teams == sorted(teams)

    def test_contains_expected_teams(self, raw_matches_df: pd.DataFrame) -> None:
        teams = get_unique_teams(raw_matches_df)
        assert "Mumbai Indians" in teams
        assert "Chennai Super Kings" in teams

    def test_no_duplicates(self, raw_matches_df: pd.DataFrame) -> None:
        teams = get_unique_teams(raw_matches_df)
        assert len(teams) == len(set(teams))

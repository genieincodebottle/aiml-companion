"""
eda.py - Exploratory Data Analysis Helpers for IPL Data
========================================================

Reusable plotting and aggregation functions for IPL match-level and
ball-by-ball datasets.  All chart functions return a Plotly ``Figure``
object so that callers can further customise or export them.

Usage
-----
>>> from src.eda import plot_season_trends, plot_team_performance
>>> fig = plot_season_trends(matches_clean)
>>> fig.show()
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (IPL-inspired)
# ---------------------------------------------------------------------------
IPL_COLORS = [
    "#004BA0",  # Mumbai Indians blue
    "#FCCA06",  # Chennai Super Kings yellow
    "#D71920",  # Royal Challengers Bangalore red
    "#3B215D",  # Kolkata Knight Riders purple
    "#FF822A",  # Sunrisers Hyderabad orange
    "#1C6EA4",  # Delhi Capitals blue
    "#EA1A85",  # Rajasthan Royals pink
    "#A72056",  # Rising Pune Supergiant magenta
    "#1DA1F2",  # Lucknow Super Giants
    "#0B4973",  # Gujarat Titans
]


# ===================================================================
# 1. Season-level trends
# ===================================================================
def plot_season_trends(
    matches_df: pd.DataFrame,
    season_col: str = "season",
) -> go.Figure:
    """Plot number of matches played per IPL season as a bar chart.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    season_col : str
        Column name holding the season identifier.

    Returns
    -------
    go.Figure
    """
    season_counts = (
        matches_df[season_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    season_counts.columns = ["Season", "Matches"]

    fig = px.bar(
        season_counts,
        x="Season",
        y="Matches",
        title="Number of IPL Matches Per Season",
        labels={"Season": "Season", "Matches": "Number of Matches"},
        color_discrete_sequence=IPL_COLORS,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def matches_per_season_table(
    matches_df: pd.DataFrame,
    season_col: str = "season",
) -> pd.DataFrame:
    """Return a summary table of matches per season.

    Returns
    -------
    pd.DataFrame
        Columns: ``Season``, ``Matches``.
    """
    return (
        matches_df.groupby(season_col)
        .size()
        .reset_index(name="Matches")
        .rename(columns={season_col: "Season"})
    )


# ===================================================================
# 2. Team performance
# ===================================================================
def compute_team_win_pct(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall win percentage for every team.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with a ``winner`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``Team``, ``Matches``, ``Wins``, ``Win_Pct``.
    """
    all_teams = pd.concat([matches_df["team1"], matches_df["team2"]])
    total = all_teams.value_counts().rename("Matches")
    wins = matches_df["winner"].value_counts().rename("Wins")
    summary = pd.concat([total, wins], axis=1).fillna(0).astype(int)
    summary["Win_Pct"] = (summary["Wins"] / summary["Matches"] * 100).round(2)
    summary = summary.reset_index().rename(columns={"index": "Team"})
    return summary.sort_values("Win_Pct", ascending=False).reset_index(drop=True)


def plot_team_performance(
    matches_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of team win percentages.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    top_n : int
        Number of teams to display.

    Returns
    -------
    go.Figure
    """
    team_stats = compute_team_win_pct(matches_df).head(top_n)

    fig = px.bar(
        team_stats,
        x="Win_Pct",
        y="Team",
        orientation="h",
        title=f"Top {top_n} IPL Teams by Win Percentage",
        labels={"Win_Pct": "Win %", "Team": ""},
        color="Win_Pct",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_team_wins_by_season(
    matches_df: pd.DataFrame,
    teams: Optional[List[str]] = None,
    season_col: str = "season",
) -> go.Figure:
    """Stacked / grouped bar chart showing wins per team per season.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    teams : list of str, optional
        Subset of teams to include.  Defaults to all.
    season_col : str
        Column holding the season identifier.

    Returns
    -------
    go.Figure
    """
    wins = matches_df.groupby([season_col, "winner"]).size().reset_index(name="Wins")
    if teams:
        wins = wins[wins["winner"].isin(teams)]

    fig = px.bar(
        wins,
        x=season_col,
        y="Wins",
        color="winner",
        title="Team Wins by Season",
        labels={season_col: "Season", "Wins": "Number of Wins", "winner": "Team"},
        barmode="group",
    )
    fig.update_layout(xaxis_tickangle=-45, legend_title_text="Team")
    return fig


# ===================================================================
# 3. Toss analysis
# ===================================================================
def compute_toss_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute toss-decision distribution and win-rate after toss win.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with ``toss_decision`` and
        ``toss_winner_won_match`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``Toss_Decision``, ``Count``, ``Pct``,
        ``Win_Rate_After_Toss_Win``.
    """
    decision_counts = matches_df["toss_decision"].value_counts()
    decision_pct = (decision_counts / decision_counts.sum() * 100).round(2)

    valid = matches_df.dropna(subset=["toss_winner_won_match"])
    win_rate = (
        valid.groupby("toss_decision")["toss_winner_won_match"].mean() * 100
    ).round(2)

    stats = pd.DataFrame(
        {
            "Toss_Decision": decision_counts.index,
            "Count": decision_counts.values,
            "Pct": decision_pct.values,
        }
    )
    stats["Win_Rate_After_Toss_Win"] = stats["Toss_Decision"].map(win_rate)
    return stats


def plot_toss_analysis(matches_df: pd.DataFrame) -> go.Figure:
    """Side-by-side charts: toss decision split and win rate after toss.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.

    Returns
    -------
    go.Figure
    """
    stats = compute_toss_stats(matches_df)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Toss Decision Distribution", "Win Rate After Winning Toss"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    # Pie chart for decision split
    fig.add_trace(
        go.Pie(
            labels=stats["Toss_Decision"],
            values=stats["Count"],
            hole=0.4,
            marker_colors=IPL_COLORS[:len(stats)],
        ),
        row=1,
        col=1,
    )

    # Bar chart for win rate
    fig.add_trace(
        go.Bar(
            x=stats["Toss_Decision"],
            y=stats["Win_Rate_After_Toss_Win"],
            marker_color=IPL_COLORS[:len(stats)],
            text=stats["Win_Rate_After_Toss_Win"].apply(lambda v: f"{v:.1f}%"),
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text="IPL Toss Analysis",
        showlegend=False,
        height=450,
    )
    return fig


# ===================================================================
# 4. Venue analysis
# ===================================================================
def plot_venue_stats(
    matches_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Horizontal bar chart of top venues by number of matches hosted.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with a ``venue`` column.
    top_n : int
        Number of venues to display.

    Returns
    -------
    go.Figure
    """
    venue_counts = matches_df["venue"].value_counts().head(top_n)

    fig = px.bar(
        x=venue_counts.values,
        y=venue_counts.index,
        orientation="h",
        title=f"Top {top_n} IPL Venues by Number of Matches",
        labels={"x": "Number of Matches", "y": "Venue"},
    )
    fig.update_yaxes(title_text="Venue")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def venue_win_pct(
    matches_df: pd.DataFrame,
    min_matches: int = 5,
) -> pd.DataFrame:
    """Win percentage per team at each venue (filtered by minimum matches).

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    min_matches : int
        Minimum number of matches at a venue-team combination to include.

    Returns
    -------
    pd.DataFrame
        Columns: ``team``, ``venue``, ``matches``, ``wins``, ``win_pct``.
    """
    records = []
    teams = set(matches_df["team1"]).union(set(matches_df["team2"]))
    venues = matches_df["venue"].unique()

    for team in teams:
        for venue in venues:
            played = matches_df[
                ((matches_df["team1"] == team) | (matches_df["team2"] == team))
                & (matches_df["venue"] == venue)
            ]
            n_matches = len(played)
            if n_matches >= min_matches:
                n_wins = (played["winner"] == team).sum()
                records.append(
                    {
                        "team": team,
                        "venue": venue,
                        "matches": n_matches,
                        "wins": n_wins,
                        "win_pct": round(n_wins / n_matches * 100, 2),
                    }
                )

    return (
        pd.DataFrame(records)
        .sort_values("win_pct", ascending=False)
        .reset_index(drop=True)
    )


# ===================================================================
# 5. Batsman performance
# ===================================================================
def top_run_scorers(
    deliveries_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Aggregate top run scorers from the deliveries dataset.

    Parameters
    ----------
    deliveries_df : pd.DataFrame
        Cleaned deliveries DataFrame with ``batter`` (or ``batsman``)
        and ``batsman_runs`` columns.
    top_n : int
        Number of top scorers to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``Batsman``, ``Runs``, ``Balls``, ``Strike_Rate``,
        ``Fours``, ``Sixes``.
    """
    batter_col = "batter" if "batter" in deliveries_df.columns else "batsman"

    agg = deliveries_df.groupby(batter_col).agg(
        Runs=("batsman_runs", "sum"),
        Balls=("batsman_runs", "count"),
        Fours=("is_four", "sum") if "is_four" in deliveries_df.columns else ("batsman_runs", lambda x: (x == 4).sum()),
        Sixes=("is_six", "sum") if "is_six" in deliveries_df.columns else ("batsman_runs", lambda x: (x == 6).sum()),
    )
    agg["Strike_Rate"] = (agg["Runs"] / agg["Balls"] * 100).round(2)
    agg = agg.sort_values("Runs", ascending=False).head(top_n).reset_index()
    agg = agg.rename(columns={batter_col: "Batsman"})
    return agg


def plot_batsman_comparison(
    deliveries_df: pd.DataFrame,
    batsmen: List[str],
    metric: str = "Runs",
) -> go.Figure:
    """Bar chart comparing selected batsmen on a given metric.

    Parameters
    ----------
    deliveries_df : pd.DataFrame
        Cleaned deliveries DataFrame.
    batsmen : list of str
        Batsman names to compare.
    metric : str
        One of ``'Runs'``, ``'Strike_Rate'``, ``'Fours'``, ``'Sixes'``.

    Returns
    -------
    go.Figure
    """
    scorers = top_run_scorers(deliveries_df, top_n=200)
    subset = scorers[scorers["Batsman"].isin(batsmen)]

    fig = px.bar(
        subset,
        x="Batsman",
        y=metric,
        title=f"Batsman Comparison — {metric}",
        color="Batsman",
        color_discrete_sequence=IPL_COLORS,
    )
    return fig


# ===================================================================
# 6. Win margin distributions
# ===================================================================
def plot_win_margin_distribution(matches_df: pd.DataFrame) -> go.Figure:
    """Box plots for win-by-runs and win-by-wickets margins.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame with ``win_by_runs`` and
        ``win_by_wickets`` columns.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=matches_df.loc[matches_df["win_by_runs"] > 0, "win_by_runs"],
            name="Win by Runs",
            marker_color=IPL_COLORS[0],
        )
    )
    fig.add_trace(
        go.Box(
            y=matches_df.loc[matches_df["win_by_wickets"] > 0, "win_by_wickets"],
            name="Win by Wickets",
            marker_color=IPL_COLORS[1],
        )
    )

    fig.update_layout(
        title="Distribution of Win Margins",
        yaxis_title="Margin",
        showlegend=True,
    )
    return fig


# ===================================================================
# 7. Player-of-the-match analysis
# ===================================================================
def plot_player_of_match(
    matches_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Bar chart of players with the most Player-of-the-Match awards.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Cleaned matches DataFrame.
    top_n : int
        Number of players to show.

    Returns
    -------
    go.Figure
    """
    pom = (
        matches_df[matches_df["player_of_match"] != "Not Awarded"]["player_of_match"]
        .value_counts()
        .head(top_n)
    )

    fig = px.bar(
        x=pom.values,
        y=pom.index,
        orientation="h",
        title=f"Top {top_n} Player of the Match Award Winners",
        labels={"x": "Awards", "y": "Player"},
        color_discrete_sequence=IPL_COLORS,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# ===================================================================
# 8. Utility: save figure
# ===================================================================
def save_figure(
    fig: go.Figure,
    filepath: str,
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> None:
    """Export a Plotly figure to a static image file.

    Requires the ``kaleido`` package for static export.
    Gracefully skips export if kaleido is not installed or export fails.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object.
    filepath : str
        Destination path (e.g. ``artifacts/figures/season_trends.png``).
    width, height : int
        Image dimensions in pixels.
    scale : int
        Resolution multiplier.
    """
    try:
        fig.write_image(filepath, width=width, height=height, scale=scale)
    except (ValueError, ImportError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not export %s: %s. Install kaleido for static image export.",
            filepath, exc,
        )

"""Cohort maturity and out-of-time splitting -- the two places where a subtle
mistake silently inflates offline scores."""
import pandas as pd
import pytest

from lapse_prediction.config import CFG
from lapse_prediction.features.labels import add_labels, mature, time_split


@pytest.mark.parametrize("days,expected", [
    (0, 0), (7, 0), (8, 1), (15, 1), (16, 2), (30, 2), (31, 3),
    (45, 3), (46, 4), (400, 4), (None, 4), (float("nan"), 4),
])
def test_bucket_boundaries(days, expected):
    assert CFG.bucket_of(days) == expected


def test_lapse_flag_matches_the_lapse_bucket(modelling_table):
    m = modelling_table
    assert (m["lapsed"] == (m["bucket"] == CFG.lapse_index)).all()


def test_maturity_filter_drops_cohorts_still_inside_grace(ledger):
    df = add_labels(pd.DataFrame(ledger))
    as_of = df["due_date"].max()
    kept = mature(df, as_of=as_of)
    assert len(kept) < len(df), "nothing was dropped -- the filter is inert"
    latest_kept = kept["due_date"].max() + pd.Timedelta(days=CFG.grace_days)
    assert latest_kept <= as_of


def test_splits_are_chronological_and_disjoint(modelling_table):
    train, test, valid = time_split(modelling_table)
    assert len(train) and len(test) and len(valid)
    assert train["due_date"].max() < test["due_date"].min()
    assert test["due_date"].max() < valid["due_date"].min()
    total = len(train) + len(test) + len(valid)
    assert total == len(modelling_table)


def test_no_policy_leaks_a_future_due_into_training(modelling_table):
    """A policy may appear in several splits (different dues) -- that is correct
    for this grain -- but every training due must predate every validation due."""
    train, _, valid = time_split(modelling_table)
    assert train["due_date"].max() < valid["due_date"].min()

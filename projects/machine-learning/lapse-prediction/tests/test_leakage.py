"""Leakage is the failure mode that makes this model look brilliant offline and
useless in production. These tests are the reason the guard exists."""
import numpy as np
import pandas as pd
import pytest

from lapse_prediction.config import CFG
from lapse_prediction.features.build import (assert_no_leakage, build,
                                             feature_columns)


def test_target_columns_never_reach_the_feature_matrix(modelling_table):
    cols = feature_columns(modelling_table)
    for forbidden in ("days_to_pay", "bucket", "lapsed"):
        assert forbidden not in cols
    assert_no_leakage(cols)


def test_assert_no_leakage_raises_on_a_leaky_column():
    with pytest.raises(ValueError, match="leaky"):
        assert_no_leakage(["policy_year", "days_to_pay"])


def test_history_features_use_only_prior_dues(ledger):
    """Mutate the LAST due's outcome; features on that row must not move.

    This is the strongest available check: if any feature is built from the
    current row's payment, changing that payment changes the feature.
    """
    pid = ledger["policy_id"].value_counts().idxmax()
    pol = ledger[ledger["policy_id"] == pid].sort_values("due_date")
    assert len(pol) >= 3

    a = build(pol.copy())
    tampered = pol.copy()
    idx = tampered.index[-1]
    tampered.loc[idx, "days_to_pay"] = 999.0     # a wildly different outcome
    b = build(tampered)

    cols = feature_columns(a)
    last_a, last_b = a.iloc[-1][cols], b.iloc[-1][cols]
    pd.testing.assert_series_equal(last_a, last_b, check_names=False)


def test_first_due_has_no_payment_history(modelling_table):
    first = modelling_table[modelling_table["prior_dues"] == 0]
    assert len(first) > 0
    assert first["days_late_lag1"].isna().all()
    assert (first["prior_lapses"] == 0).all()

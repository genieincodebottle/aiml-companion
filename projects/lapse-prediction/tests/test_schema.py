"""The ingest contract."""
import pandas as pd
import pytest

from lapse_prediction.data import schema


def test_valid_ledger_passes(ledger):
    assert schema.validate(ledger) is ledger


def test_missing_column_is_rejected(ledger):
    with pytest.raises(schema.SchemaError, match="missing required columns"):
        schema.validate(ledger.drop(columns=["payment_mode"]))


def test_duplicate_grain_is_rejected(ledger):
    dupe = pd.concat([ledger, ledger.head(1)], ignore_index=True)
    with pytest.raises(schema.SchemaError, match="duplicate"):
        schema.validate(dupe)


def test_negative_days_to_pay_is_rejected(ledger):
    bad = ledger.copy()
    bad.loc[bad.index[0], "days_to_pay"] = -3.0
    with pytest.raises(schema.SchemaError, match="negative"):
        schema.validate(bad)


def test_summarise_reports_the_grain(ledger):
    s = schema.summarise(ledger)
    assert s["rows"] == len(ledger)
    assert s["policies"] == ledger["policy_id"].nunique()
    assert 0.0 <= s["never_paid_rate"] <= 1.0

"""Persistence carries provenance, and serving refuses to score with a model
whose features no longer exist."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from lapse_prediction.config import CFG
from lapse_prediction.features.labels import time_split
from lapse_prediction.models import registry
from lapse_prediction.models.ordinal import OrdinalChain
from lapse_prediction.serving import decide


@pytest.fixture(scope="module")
def fitted(modelling_table):
    train, test, _ = time_split(modelling_table)
    return OrdinalChain().fit(train, valid=test), train


def test_save_and_load_round_trip(fitted, modelling_table, tmp_path):
    model, train = fitted
    card = registry.build_card("test_model", model, train, metrics={"lapse_pr_auc": 0.4})
    registry.save(model, card, tmp_path)

    loaded, loaded_card = registry.load("test_model", "latest", tmp_path)
    assert loaded_card["version"] == card.version
    assert loaded_card["metrics"]["lapse_pr_auc"] == 0.4
    np.testing.assert_allclose(
        model.predict_proba(modelling_table.head(50)),
        loaded.predict_proba(modelling_table.head(50)))


def test_card_records_provenance(fitted):
    model, train = fitted
    card = registry.build_card("m", model, train)
    d = json.loads(card.to_json())
    for k in ("git_sha", "created_utc", "data_fingerprint", "features",
              "train_date_min", "train_date_max", "config"):
        assert d[k], f"model card missing {k}"
    assert d["n_train_rows"] == len(train)


def test_fingerprint_changes_when_data_changes(fitted):
    _, train = fitted
    cols = ["policy_id", "due_date", "days_to_pay"]
    a = registry.data_fingerprint(train[cols])
    tampered = train.copy()
    tampered.loc[tampered.index[0], "days_to_pay"] = 42.0
    assert a != registry.data_fingerprint(tampered[cols])


def test_load_unknown_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.load("does_not_exist", "latest", tmp_path)


def test_queue_is_ranked_by_premium_at_risk(fitted, modelling_table):
    model, _ = fitted
    sample = modelling_table.tail(200)
    q = decide.score(sample, model.predict_proba(sample), CFG)
    assert q["priority_score"].is_monotonic_decreasing
    assert set(q["action"]) <= {"call", "monitor"}
    assert (q["contact_on_day"] >= 0).all()
    assert (q["contact_on_day"] < CFG.grace_days).all()
    # the queue comes back sorted, so join on the grain rather than by position
    joined = q.merge(sample[["policy_id", "due_date", "annual_premium"]],
                     on=["policy_id", "due_date"], how="left")
    np.testing.assert_allclose(
        joined["premium_at_risk"], joined["annual_premium"] * joined["p_lapse"],
        rtol=1e-9)


def test_queue_respects_calling_capacity(fitted, modelling_table):
    model, _ = fitted
    sample = modelling_table.tail(400)
    q = decide.score(sample, model.predict_proba(sample), CFG, capacity_pct=0.10)
    share = (q["action"] == "call").mean()
    assert 0.05 <= share <= 0.15, f"capacity honoured badly: {share:.3f}"


def test_value_of_queue_reports_money(fitted, modelling_table):
    model, _ = fitted
    sample = modelling_table.tail(200)
    v = decide.value_of_queue(decide.score(sample, model.predict_proba(sample), CFG))
    assert v["policies_called"] > 0
    assert 0 <= v["share_of_total_risk_covered"] <= 1
    assert v["expected_premium_saved"] > 0

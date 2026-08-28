"""End-to-end: the pipelines must run, gate, persist and score for real."""
import dataclasses
from pathlib import Path

import pytest

from lapse_prediction.config import CFG
from lapse_prediction.evaluation.report import gate
from lapse_prediction.pipelines import benchmark, train
from lapse_prediction.serving.predict import Scorer


@pytest.fixture(scope="module")
def tmp_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    return dataclasses.replace(
        CFG, raw_data=str(d / "dues.parquet"),
        modelling_table=str(d / "modelling.parquet"),
        model_store=str(d / "models"), artifacts=str(d / "artifacts"))


@pytest.mark.slow
def test_train_persists_a_model_that_passes_the_gate(tmp_cfg):
    out = train.run("ordinal_chain", tmp_cfg, n_policies=3000, persist=True)
    head = out["metrics"]["headline"]

    assert head["n_valid"] > 0
    assert 0 < head["lapse_rate"] < 0.5
    assert head["monotonicity_violation"] == 0.0
    ok, fails = gate(head)
    assert ok, f"release gate failed: {fails}"

    from lapse_prediction.models.registry import list_versions
    assert list_versions("ordinal_chain", tmp_cfg.model_store)


@pytest.mark.slow
def test_scorer_writes_a_retention_queue(tmp_cfg):
    train.run("ordinal_chain", tmp_cfg, n_policies=3000, persist=True)
    q = Scorer("ordinal_chain", "latest", tmp_cfg).score_batch()
    assert len(q) > 0
    assert q["model_version"].nunique() == 1
    assert {"p_lapse", "expected_days_if_paid", "contact_on_day",
            "action"} <= set(q.columns)
    assert (Path(tmp_cfg.artifacts) / "retention_queue.csv").exists()


@pytest.mark.slow
def test_benchmark_ranks_every_family(tmp_cfg):
    res = benchmark.run(tmp_cfg, n_policies=2000,
                        only={"prior", "logit", "ordinal_chain"})
    assert len(res) >= 3
    assert res["lapse_pr_auc"].is_monotonic_decreasing
    prior = res.loc[res["model"] == "prior", "lapse_pr_auc"].iloc[0]
    best = res["lapse_pr_auc"].max()
    assert best > prior, "no model beat the prior baseline"


def test_gate_rejects_a_bad_model():
    ok, fails = gate({"lapse_pr_auc": 0.05, "lapse_brier": 0.5,
                      "monotonicity_violation": 0.3})
    assert not ok and len(fails) == 3

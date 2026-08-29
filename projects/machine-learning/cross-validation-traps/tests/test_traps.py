"""Each trap must reproduce its own direction. Sizes are checked loosely."""
import dataclasses

import numpy as np

from cv_traps.evaluation.metrics import optimism_table
from cv_traps.pipelines import traps


def _small(cfg):
    """Enough rows for the effects to survive, few enough to run in tests."""
    return dataclasses.replace(cfg, n_customers=1400, n_noise=60,
                               n_repeats=5, n_candidates=12,
                               preprocess_sizes=(300, 1500)).validate()


def test_optimism_is_cv_minus_truth():
    table = optimism_table([{"scheme": "a", "cv_auc": 0.80},
                            {"scheme": "b", "cv_auc": 0.70}], truth=0.75)
    assert list(table["optimism"]) == [0.05, -0.05]


def test_ungrouped_folds_score_higher_than_grouped(cfg, monkeypatch, tmp_path):
    """The core claim of the project, in one assertion."""
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    out = traps.run_grouped(small, write=False)
    t = out["table"].set_index("scheme")
    assert t.loc["stratified_kfold", "cv_auc"] > t.loc["group_kfold", "cv_auc"]
    assert t.loc["stratified_kfold", "pct_test_rows_seen_customer"] > 0.8
    assert t.loc["group_kfold", "pct_test_rows_seen_customer"] == 0.0


def test_forward_chaining_removes_the_future(cfg, tmp_path):
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    t = traps.run_temporal(small, write=False)["table"].set_index("scheme")
    assert t.loc["stratified_kfold", "train_rows_from_the_future"] > 0
    assert t.loc["forward_chaining", "train_rows_from_the_future"] == 0
    assert t.loc["grouped_forward_chaining", "train_rows_from_the_future"] == 0


def test_selection_leakage_shrinks_as_the_sample_grows(cfg, tmp_path):
    """The finding that reframed this trap, pinned so it cannot rot.

    Selecting features on all the data before validating is nearly free at
    thousands of rows and severe at hundreds. Quoting a single number for it
    would be quoting a number that depends on a variable nobody mentions.
    """
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    curve = traps.run_preprocessing(small, write=False)["curve"]
    assert len(curve) >= 2
    assert curve.iloc[0]["self_deception"] > curve.iloc[-1]["self_deception"]
    assert curve.iloc[0]["self_deception"] > 0.02


def test_the_winner_beats_the_average_candidate(cfg, tmp_path):
    """Candidates are interchangeable by construction, so any gap is luck."""
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    out = traps.run_selection(small, write=False)
    t = out["table"].set_index("estimate")
    assert (t.loc["best candidate's own CV score", "cv_auc"]
            >= t.loc["mean across all candidates", "cv_auc"])
    assert out["spread"] > 0


def test_repeated_cv_actually_varies(cfg, tmp_path):
    """Regression test for the deterministic-splitter bug."""
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    table = traps.run_variance(small, write=False)["table"]
    assert (table["std_across_repeats"] > 0).all(), (
        "a spread of exactly zero means the splits never changed")


def test_sweep_reports_every_trap(cfg, tmp_path):
    small = dataclasses.replace(_small(cfg),
                                raw_path=str(tmp_path / "p.parquet"))
    sweep = traps.run_sweep(small, write=False)
    assert set(sweep["trap"]) == {"preprocessing", "grouped", "temporal",
                                  "selection"}
    assert not sweep[["naive_cv", "careful_cv", "truth"]].isna().any().any()
    h = traps.headline(sweep)
    assert h["worst_trap"] in set(sweep["trap"])
    assert "self-deception" in h["verdict"]


def test_the_control_dials_reach_every_subcommand():
    """The README's falsifiability step, which used to be impossible.

    It said to build a control panel with `data --group-effect 0.0` and then
    run a trap. The trap read conf/config.yaml, found a cache whose manifest
    disagreed with it, correctly called it stale, and rebuilt the NON-control
    panel underneath the reader. The documented control silently measured the
    thing it was supposed to switch off.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "run.py", "grouped", "--group-effect", "0.0",
         "--drift", "0.0", "--no-write"],
        cwd=root, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr
    assert "--group-effect" not in out.stderr
    assert "group_effect=0.0" in out.stdout, (
        "the control override never reached the pipeline")


def test_with_no_latent_effect_grouping_buys_nothing(exchangeable_panel, cfg):
    """The control, as a measurement rather than as a CLI flag.

    If grouped folds simply scored lower because they train on fewer distinct
    customers, this would fail, and the project's central claim would be an
    artefact of the method rather than a property of the data.
    """
    from cv_traps.evaluation.metrics import cv_score
    from cv_traps.evaluation.truth import split_panel
    from cv_traps.features.build import booster, xy
    from cv_traps.splitters import schemes

    dev, _ = split_panel(exchangeable_panel, cfg)
    X, y = xy(dev, core_only=True)
    model = booster(seed=cfg.seed)
    ungrouped, _ = cv_score(model, X, y,
                            list(schemes.stratified(dev, y, cfg.n_folds, cfg.seed)))
    grouped, _ = cv_score(model, X, y,
                          list(schemes.grouped(dev, y, cfg.n_folds, cfg.seed)))
    assert abs(ungrouped - grouped) < 0.02, (
        f"control failed: ungrouped={ungrouped:.4f} grouped={grouped:.4f}; "
        "the gap should vanish when there is no latent customer effect")

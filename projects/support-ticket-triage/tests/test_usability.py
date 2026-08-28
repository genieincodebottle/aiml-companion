"""Things a beginner hits in the first ten minutes."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from support_ticket_triage.data import schema
from support_ticket_triage.evaluation.metrics import (
    expected_calibration_error, multiclass_brier)
from support_ticket_triage.features.build import feature_columns, split, xy

ROOT = Path(__file__).resolve().parents[1]


def test_run_py_exists_and_needs_no_install():
    assert (ROOT / "run.py").exists(), (
        "run.py is the zero-install entry point the README promises")


def test_cli_help_works_without_installing_the_package():
    out = subprocess.run([sys.executable, "run.py", "--help"], cwd=ROOT,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    for cmd in ("independence", "strategies", "calibrate", "sweep", "route"):
        assert cmd in out.stdout, f"{cmd} is missing from the help text"


def test_no_write_flag_works_after_the_subcommand(tmp_path):
    """Argparse puts global flags before the verb; people type them after.

    Runs against a throwaway config. The CLI generates and caches an inbox on
    first use, and a test that lets it write into the project's real data/
    would leave a cache behind for anyone who runs pytest before the pipeline.
    """
    conf = tmp_path / "config.yaml"
    conf.write_text("\n".join([
        "data:",
        "  n_tickets: 1200",
        f"  raw_path: {(tmp_path / 'tickets.parquet').as_posix()}",
        "paths:",
        f"  artifacts: {(tmp_path / 'artifacts').as_posix()}",
    ]), encoding="utf-8")
    env = {**os.environ, "TRIAGE_CONFIG": str(conf)}

    # Snapshot the project cache first. Asserting the directory does not EXIST
    # would fail for anyone who has run the pipeline once, which is everyone;
    # the property that actually matters is that this run did not touch it.
    project_data = ROOT / "data"
    before = ({p.name: p.stat().st_mtime_ns for p in project_data.iterdir()}
              if project_data.exists() else None)

    out = subprocess.run(
        [sys.executable, "run.py", "independence", "--no-write"],
        cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
    assert out.returncode == 0, out.stderr

    assert (tmp_path / "tickets.parquet").exists(), (
        "TRIAGE_CONFIG was ignored: nothing was written to the temp path")
    after = ({p.name: p.stat().st_mtime_ns for p in project_data.iterdir()}
             if project_data.exists() else None)
    assert after == before, (
        "the CLI wrote into the project data directory despite TRIAGE_CONFIG")


def test_split_keeps_the_rare_class_in_both_halves(tickets, cfg):
    train, test = split(tickets, cfg)
    for part, name in ((train, "train"), (test, "test")):
        counts = part["category"].value_counts()
        assert len(counts) == 6, f"{name} lost a class entirely"
        assert counts.min() >= 5, (
            f"{name} has only {counts.min()} rows of the rarest class")


def test_leaky_columns_cannot_reach_the_model(tickets):
    X, _ = xy(tickets)
    for bad in ("category", "text", "ticket_id"):
        assert bad not in X.columns


def test_schema_rejects_a_broken_export(tickets):
    with pytest.raises(schema.SchemaError, match="not unique"):
        bad = tickets.copy()
        bad.loc[1, "ticket_id"] = bad.loc[0, "ticket_id"]
        schema.validate(bad)

    with pytest.raises(schema.SchemaError, match="unknown categories"):
        bad = tickets.copy()
        bad.loc[0, "category"] = "not_a_real_class"
        schema.validate(bad)

    with pytest.raises(schema.SchemaError, match="binary"):
        bad = tickets.copy()
        col = feature_columns(bad)[0]
        bad.loc[0, col] = 7
        schema.validate(bad)


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    """Sanity-check the metric itself before trusting it about a model."""
    rng = np.random.default_rng(0)
    classes = np.array(["a", "b"])
    conf = rng.uniform(0.5, 1.0, 20000)
    proba = np.column_stack([conf, 1 - conf])
    # make it right exactly `conf` of the time, which is what calibrated means
    y = np.where(rng.random(20000) < conf, "a", "b")
    assert expected_calibration_error(y, proba, classes, 10) < 0.02


def test_brier_rewards_honesty_over_confidence():
    classes = np.array(["a", "b"])
    y = np.array(["a", "b"])
    honest = np.array([[0.6, 0.4], [0.4, 0.6]])
    overconfident = np.array([[0.99, 0.01], [0.01, 0.99]])
    wrong_and_sure = np.array([[0.01, 0.99], [0.99, 0.01]])
    # right and sure beats right and hedged
    assert multiclass_brier(y, overconfident, classes) < multiclass_brier(
        y, honest, classes)
    # but being sure and wrong is the worst outcome of the three
    assert multiclass_brier(y, wrong_and_sure, classes) > multiclass_brier(
        y, honest, classes)


def test_empty_ticket_does_not_crash_the_pipeline(fitted, cfg):
    from support_ticket_triage.models.strategies import build
    model = build("native_nb", cfg).fit(fitted["Xtr"], fitted["ytr"])
    blank = pd.DataFrame(0.0, index=[0], columns=fitted["Xte"].columns)
    proba = model.predict_proba(blank)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

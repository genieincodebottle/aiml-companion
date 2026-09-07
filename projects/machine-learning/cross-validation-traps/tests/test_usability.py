"""Things a beginner hits in the first ten minutes."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cv_traps.data import schema
from cv_traps.evaluation.truth import split_panel

ROOT = Path(__file__).resolve().parents[1]


def test_run_py_exists_and_needs_no_install():
    assert (ROOT / "run.py").exists(), (
        "run.py is the zero-install entry point the README promises")


def test_cli_help_works_without_installing_the_package():
    out = subprocess.run([sys.executable, "run.py", "--help"], cwd=ROOT,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    for cmd in ("truth", "preprocessing", "grouped", "temporal", "selection",
                "variance", "sweep"):
        assert cmd in out.stdout, f"{cmd} is missing from the help text"


def test_no_write_flag_works_after_the_subcommand(tmp_path):
    """Argparse puts global flags before the verb; people type them after.

    Runs against a throwaway config. The CLI generates and caches a panel on
    first use, and a test that let it write into the project's real data/ would
    leave a cache behind for anyone who runs pytest before the pipeline.
    """
    conf = tmp_path / "config.yaml"
    conf.write_text("\n".join([
        "data:",
        "  n_customers: 400",
        # Must stay above select_k (20), which the config validator enforces.
        "  n_noise: 40",
        f"  raw_path: {(tmp_path / 'panel.parquet').as_posix()}",
        "paths:",
        f"  artifacts: {(tmp_path / 'artifacts').as_posix()}",
    ]), encoding="utf-8")
    env = {**os.environ, "CVTRAPS_CONFIG": str(conf)}

    # Snapshot the project cache first. Asserting the directory does not EXIST
    # would fail for anyone who has run the pipeline once, which is everyone;
    # the property that actually matters is that this run did not touch it.
    project_data = ROOT / "data"
    before = ({p.name: p.stat().st_mtime_ns for p in project_data.iterdir()}
              if project_data.exists() else None)

    out = subprocess.run([sys.executable, "run.py", "truth", "--no-write"],
                         cwd=ROOT, capture_output=True, text=True,
                         timeout=600, env=env)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "panel.parquet").exists(), (
        "CVTRAPS_CONFIG was ignored: nothing was written to the temp path")

    after = ({p.name: p.stat().st_mtime_ns for p in project_data.iterdir()}
             if project_data.exists() else None)
    assert after == before, (
        "the CLI wrote into the project data directory despite CVTRAPS_CONFIG")


def test_the_truth_holdout_shares_no_customer_with_development(split):
    assert not (set(split["dev"]["customer_id"])
                & set(split["out"]["customer_id"]))


def test_the_truth_holdout_is_strictly_in_the_future(split):
    assert split["dev"]["period"].max() < split["out"]["period"].min()


def test_a_holdout_that_would_be_empty_fails_loudly(panel, cfg):
    import dataclasses
    broken = dataclasses.replace(cfg, holdout_customer_share=0.99)
    with pytest.raises((ValueError, AssertionError)):
        split_panel(panel.head(300), broken)


def test_schema_rejects_a_broken_export(panel):
    with pytest.raises(schema.SchemaError, match="not unique"):
        bad = panel.copy()
        bad.loc[1, "row_id"] = bad.loc[0, "row_id"]
        schema.validate(bad)

    with pytest.raises(schema.SchemaError, match="varies within a customer"):
        bad = panel.copy()
        bad.loc[0, "plan_price"] = 999.0
        schema.validate(bad)

    with pytest.raises(schema.SchemaError, match="binary"):
        bad = panel.copy()
        bad.loc[0, "churned"] = 7
        schema.validate(bad)


def test_leaky_columns_cannot_reach_the_model(panel):
    from cv_traps.features.build import xy
    X, _ = xy(panel)
    for bad in ("customer_id", "period", "row_id", "churned"):
        assert bad not in X.columns, (
            f"{bad} reached the design matrix; customer_id in particular IS "
            "the group trap in its most literal form")


def test_auc_returns_nan_rather_than_raising_on_a_single_class_fold():
    """Forward-chaining folds are period-sized and can arrive single-class."""
    import numpy as np

    from cv_traps.evaluation.metrics import auc
    assert np.isnan(auc(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])))


def test_a_typod_config_path_refuses_instead_of_using_defaults(tmp_path):
    """The quietest failure this project had.

    A wrong --config or CVTRAPS_CONFIG used to fall back to the built-in
    defaults without a word, so a run against a 300-customer config silently
    used the full 2,600-customer panel and every printed number belonged to a
    different experiment than the one the operator thought they had started.
    """
    for env, args in (
            ({}, ["--config", str(tmp_path / "nope.yaml"), "truth"]),
            ({"CVTRAPS_CONFIG": str(tmp_path / "nope.yaml")}, ["truth"])):
        out = subprocess.run([sys.executable, "run.py", *args], cwd=ROOT,
                             capture_output=True, text=True, timeout=300,
                             env={**os.environ, **env})
        assert out.returncode != 0, "a missing config file must not be ignored"
        assert "config file not found" in out.stderr
        assert "Traceback" not in out.stderr, (
            "a bad setting is an ordinary mistake, not a crash")


def test_bad_settings_print_a_message_rather_than_a_traceback(tmp_path):
    conf = tmp_path / "config.yaml"
    conf.write_text("data:\n  n_customers: 400\n  n_noise: 40\n", encoding="utf-8")
    env = {**os.environ, "CVTRAPS_CONFIG": str(conf)}
    for args, expect in ((["data", "--n", "50"], "too small"),
                         (["data", "--group-effect", "-1"], "magnitudes"),
                         (["--log-level", "LOUD", "truth"], "unknown log level")):
        out = subprocess.run([sys.executable, "run.py", *args], cwd=ROOT,
                             capture_output=True, text=True, timeout=300, env=env)
        assert out.returncode == 2, f"{args} should exit 2"
        assert expect in out.stderr, f"{args}: {out.stderr[:200]}"
        assert "Traceback" not in out.stderr


def test_log_level_is_not_silently_ignored():
    """Every module calls get_logger at import, which used to lock the level.

    `configure()` returned early once anything had imported, so --log-level was
    documented in --help and did nothing at all.
    """
    import logging

    from cv_traps.utils.logging import configure
    try:
        configure("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        configure("WARNING")
        assert logging.getLogger().level == logging.WARNING
    finally:
        configure("INFO")

    with pytest.raises(ValueError, match="unknown log level"):
        configure("LOUD")


def test_an_unreadable_cache_regenerates_instead_of_raising(tmp_path, cfg):
    """An interrupted first run leaves a partial file. It is only a cache."""
    import dataclasses

    from cv_traps.data.io import load_panel
    small = dataclasses.replace(cfg, n_customers=300, n_noise=30,
                                raw_path=str(tmp_path / "panel.parquet"))
    load_panel(cfg=small, refresh=True)
    (tmp_path / "panel.parquet").write_bytes(b"truncated")
    df = load_panel(cfg=small)
    assert df["customer_id"].nunique() == 300

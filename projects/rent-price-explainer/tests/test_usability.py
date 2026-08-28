"""Beginner-friction regressions.

Each test here corresponds to something that actually broke a cold start.
They are cheap, and they stop the project quietly becoming unrunnable.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from rent_price_explainer.config import CFG
from rent_price_explainer.data.io import _staleness, load_listings

ROOT = Path(__file__).resolve().parents[1]


def test_run_py_exists_and_needs_no_install():
    """The zero-install entry point is the documented happy path."""
    assert (ROOT / "run.py").exists()
    out = subprocess.run([sys.executable, str(ROOT / "run.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "diagnose" in out.stdout


def test_diagnose_exits_zero_by_default():
    """The naive model is SUPPOSED to fail its checks; that must not look like
    a broken build to make/CI."""
    out = subprocess.run([sys.executable, str(ROOT / "run.py"), "diagnose"],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert out.returncode == 0, out.stderr[-800:]
    assert "FAILED" in out.stdout          # it still reports the failures


def test_stale_cache_is_detected(tmp_path):
    """Silently returning data that does not match the config is the worst
    failure mode: nothing errors, the numbers are just wrong."""
    import pandas as pd
    cfg = dataclasses.replace(CFG, n_listings=400, junk_features=3,
                              raw_path=str(tmp_path / "l.parquet")).validate()
    first = load_listings(cfg=cfg, validate=False)
    assert len(first) == 400

    bigger = dataclasses.replace(cfg, n_listings=900).validate()
    second = load_listings(cfg=bigger, validate=False)
    assert len(second) == 900, "stale cache was served instead of regenerating"

    fewer_junk = dataclasses.replace(bigger, junk_features=6).validate()
    third = load_listings(cfg=fewer_junk, validate=False)
    assert len([c for c in third.columns if c.startswith("junk_")]) == 6


def test_staleness_reports_the_reason():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "junk_1": [0.1, 0.2]})
    reasons = _staleness(df, dataclasses.replace(CFG, n_listings=99,
                                                 junk_features=4))
    assert len(reasons) == 2 and "rows" in reasons[0]


def test_compare_survives_without_shap(monkeypatch):
    """shap is a heavy optional dependency. Losing it must cost the attribution
    table, not the six sections of analysis above it."""
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "shap":
            raise ImportError("No module named shap")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from rent_price_explainer.pipelines import compare
    out = compare.run(dataclasses.replace(CFG, n_listings=800).validate(),
                      write=False)
    assert "accuracy" in out and "recovery" in out
    assert "attribution" not in out


def test_notebook_is_standalone():
    """It must import nothing from this package and touch no files."""
    import json
    nb = json.loads((ROOT / "notebooks" /
                     "rent_price_explainer_standalone.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    assert "rent_price_explainer" not in code, "notebook depends on the package"
    for bad in ("to_csv(", "to_parquet(", "read_csv(", "read_parquet("):
        assert bad not in code, f"notebook touches the filesystem via {bad}"


def test_cache_is_actually_reused(tmp_path, caplog):
    """A cache that silently regenerates every run is not a cache. This caught
    a real bug: a NameError inside the corrupt-file handler made every run
    look like a corrupted cache."""
    import logging
    cfg = dataclasses.replace(CFG, n_listings=300, junk_features=2,
                              raw_path=str(tmp_path / "l.parquet")).validate()
    load_listings(cfg=cfg, validate=False)                    # builds it
    caplog.clear()          # caplog accumulates across the whole test
    with caplog.at_level(logging.INFO):
        load_listings(cfg=cfg, validate=False)                # must reuse it
    text = caplog.text
    assert "loaded listings from" in text
    assert "generating synthetic market" not in text


def test_corrupt_cache_recovers_without_a_traceback(tmp_path):
    """An interrupted first run leaves a half-written file; that must not
    surface as a pyarrow stack trace."""
    p = tmp_path / "l.parquet"
    p.write_text("this is not a parquet file")
    cfg = dataclasses.replace(CFG, n_listings=300, junk_features=2,
                              raw_path=str(p)).validate()
    df = load_listings(cfg=cfg, validate=False)
    assert len(df) == 300

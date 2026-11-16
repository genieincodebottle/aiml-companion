"""GATE 5: diagnose the three recorded incidents from the data.

Write debug_lab/diagnoses/a_kv_saturation.md, b_hot_prefix_imbalance.md and
c_speculation_collapse.md in the format described in debug_lab/check_diagnosis.py. The
checker verifies the NAMED MECHANISM, that every number you cite is really in the scrape you
point at, that a counter is read from two scrapes, and that the FIX names the metric that
would confirm it. It does not grade your writing.

A fourth incident, d_config_regression (a before/after pair with a config diff), is
optional: it is checked only when you write debug_lab/diagnoses/d_config_regression.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug_lab"))
import check_diagnosis as cd  # noqa: E402

REQUIRED = sorted(set(cd.SPEC) - cd.OPTIONAL)


@pytest.mark.parametrize("incident", REQUIRED)
def test_diagnosis(incident):
    res = cd.check(incident)
    assert res.ok, f"{incident}: " + "; ".join(res.reasons)


@pytest.mark.parametrize("incident", sorted(cd.OPTIONAL))
def test_optional_diagnosis(incident):
    if not (cd.ROOT / "diagnoses" / f"{incident}.md").exists():
        pytest.skip(f"{incident} is optional: write debug_lab/diagnoses/{incident}.md to be checked")
    res = cd.check(incident)
    assert res.ok, f"{incident}: " + "; ".join(res.reasons)

"""GATE 5: diagnose the three recorded incidents from the data.

Write debug_lab/diagnoses/a_kv_saturation.md, b_hot_prefix_imbalance.md and
c_speculation_collapse.md in the format described in debug_lab/check_diagnosis.py. The
checker verifies the NAMED MECHANISM and that every number you cite is really in the
scrape you point at. It does not grade your writing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug_lab"))
import check_diagnosis as cd  # noqa: E402


@pytest.mark.parametrize("incident", sorted(cd.SPEC))
def test_diagnosis(incident):
    res = cd.check(incident)
    assert res.ok, f"{incident}: " + "; ".join(res.reasons)

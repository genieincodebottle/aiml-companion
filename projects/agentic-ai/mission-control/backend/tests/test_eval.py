"""The eval plane's promotion gate."""

from __future__ import annotations

from evaluation.run_eval import main as eval_main


def test_eval_promotes_in_stub_mode():
    # Deterministic stub investigations should clear the gate: 100% disposition
    # accuracy and judge score above threshold -> exit 0 (PROMOTE).
    assert eval_main([]) == 0

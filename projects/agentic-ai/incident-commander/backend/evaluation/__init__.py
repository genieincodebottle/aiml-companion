"""Evaluation: outcome, efficiency, safety, and process-quality metrics.

Read `metrics.py` first - it explains WHY each metric exists, which matters more
than the numbers themselves. `judge.py` is the LLM-as-judge layer that scores
reasoning rather than outcomes. `run_eval.py` is the runner.
"""

from .metrics import aggregate, compare_to_baseline, score_trajectory, wilson_interval

__all__ = ["aggregate", "compare_to_baseline", "score_trajectory", "wilson_interval"]

"""Evaluation harness for the Migration Engineer.

Runs the datetime-fleet job through the LangGraph pipeline, scores the run against the
golden expectations and fleet thresholds, prints a scorecard, and exits non-zero on any
regression - so it doubles as a promotion gate in CI/CD.

Run:  uv run python -m evaluation.run_eval

Deterministic and free in the default (stub) mode. Set ANTHROPIC_API_KEY + install the
SDK to evaluate the live Claude Agent SDK worker instead (this consumes model quota).
"""

from __future__ import annotations

import asyncio
import sys

from evaluation.golden import GOLDEN, MAX_THRESHOLDS, THRESHOLDS
from evaluation.scorers import check_golden, score_fleet
from src.config import get_settings
from src.orchestrator.graph import run_batch
from src.rulebook.rules import get_job


async def main() -> int:
    settings = get_settings()
    job = get_job("datetime-fleet")
    summary = await run_batch(job, approval_policy="auto")
    scores = score_fleet(summary)
    violations = check_golden(summary, GOLDEN)

    print(f"\n=== Migration Engineer Evaluation  (mode={settings.execution_mode}) ===\n")
    print(f"{'repo':<26}{'outcome':<18}{'tests':<8}{'files':<7}{'steps':<6}tampered")
    print("-" * 78)
    for r in summary["repos"]:
        print(
            f"{r['name']:<26}{str(r['outcome']):<18}{str(r['tests_passing']):<8}"
            f"{len(r['files_changed']):<7}{r['steps']:<6}{r['tampered_with_tests']}"
        )
    print("-" * 78)
    # Headline first: passed AND did not touch the tests. The two component
    # rates below can each look perfect while a repo cheated -- see score_fleet.
    print(f"Clean migration rate   : {scores['clean_migration_rate']:.0%}  <- headline")
    print(f"  tests passing        : {scores['migration_success_rate']:.0%}")
    print(f"  tests untouched      : {scores['no_tamper_rate']:.0%}  (anti reward-hacking)")
    if scores["hacked_pass_rate"]:
        print(f"  PASSED BY CHEATING   : {scores['hacked_pass_rate']:.0%}  <- reward hacking detected")
    print(f"PR-open rate           : {scores['pr_open_rate']:.0%}")
    print(f"Mean steps / repo      : {scores['mean_steps']}")
    # Label the cost by mode. In stub mode nothing is billed: the figure is
    # simulated usage (_SIM_PRICE_PER_1M in tools/registry.py) that exists so the
    # budget ceiling has something to meter offline. Printed unqualified, it
    # invites a learner to quote an invented dollar figure as a measurement.
    if settings.execution_mode == "stub":
        print(f"Simulated cost (USD)   : {scores['total_cost_usd']}  "
              f"(stub mode: nothing was billed)\n")
    else:
        print(f"Total cost (USD)       : {scores['total_cost_usd']}\n")

    failures: list[str] = list(violations)
    for metric, threshold in THRESHOLDS.items():
        if scores.get(metric, 0.0) < threshold:
            failures.append(f"threshold: {metric} {scores.get(metric):.2f} < {threshold:.2f}")

    # Ceilings are checked separately and in the other direction. Folding them
    # into THRESHOLDS would leave them permanently satisfied.
    for metric, ceiling in MAX_THRESHOLDS.items():
        value = scores.get(metric, 0.0)
        if value > ceiling:
            failures.append(f"ceiling: {metric} {value:.2f} > {ceiling:.2f}")

    if failures:
        print("EVAL FAILED:")
        for f in failures:
            print(f"  - {f}")
        print()
        return 1
    print("EVAL PASSED: all golden expectations and thresholds met.\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

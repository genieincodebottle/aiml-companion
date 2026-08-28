#!/usr/bin/env python3
"""Run the Northwind Freight evaluation gate.

This runner WORKS on a fresh clone. It is supposed to FAIL on a fresh clone.
Those are different statements and the difference matters: a harness that errors
tells you nothing, a harness that runs and reports 12 cases against a floor of
30 tells you exactly what to do next.

Usage:
    python eval/run_eval.py                # heuristic judge, offline
    python eval/run_eval.py --llm          # try LLMJudge, fall back loudly
    python eval/run_eval.py --json         # machine-readable summary as well

Exit codes:
    0  every gate passed
    1  at least one gate failed (the normal state until you do the work)
    2  the harness itself could not run (missing or malformed files)

Dependencies: stdlib + pyyaml. Nothing else, on purpose. The gate has to run in
a customer environment where you do not control the package list.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not a gate failure
    print("FATAL: pyyaml is not installed. Run: pip install -r requirements.txt")
    sys.exit(2)

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.judge import FAILURE_MODES, get_judge, validate_modes  # noqa: E402

GOLDEN_SET_PATH = EVAL_DIR / "golden_set.yaml"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.yaml"


# ---------------------------------------------------------------------------
# The system under evaluation
# ---------------------------------------------------------------------------


def load_system():
    """Return a callable f(ticket_text) -> answer string.

    On a fresh clone src/ is a skeleton, so there is nothing to call. Rather
    than crash, we substitute a null system that answers nothing. Every quality
    number then reads 0.00, which is the honest score for a system that does not
    exist yet, and the coverage gates still fire first.

    Wire your pipeline in by exposing `answer(ticket: str) -> str` from
    src/pipeline.py. The runner picks it up with no changes here.
    """
    try:
        from src.pipeline import answer  # type: ignore
        return answer, "src.pipeline.answer"
    except Exception:
        return (lambda ticket: ""), "null-system (src/pipeline.py not implemented)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    if not path.exists():
        print(f"FATAL: missing {path}")
        sys.exit(2)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"FATAL: {path} is not valid YAML: {exc}")
        sys.exit(2)
    if not isinstance(data, dict):
        print(f"FATAL: {path} must contain a mapping at the top level")
        sys.exit(2)
    return data


REQUIRED_CASE_FIELDS = ("id", "failure_mode", "input", "expected", "notes")


def load_cases() -> list[dict]:
    data = load_yaml(GOLDEN_SET_PATH)
    cases = data.get("cases") or []
    if not isinstance(cases, list):
        print("FATAL: golden_set.yaml 'cases' must be a list")
        sys.exit(2)

    problems: list[str] = []
    seen: set[str] = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            problems.append(f"case #{i} is not a mapping")
            continue
        for fieldname in REQUIRED_CASE_FIELDS:
            if not case.get(fieldname):
                problems.append(f"case #{i} ({case.get('id', '?')}) missing '{fieldname}'")
        cid = str(case.get("id", ""))
        if cid in seen:
            problems.append(f"duplicate case id '{cid}' - ids must be stable and unique")
        seen.add(cid)

    unknown = validate_modes(str(c.get("failure_mode", "")) for c in cases if isinstance(c, dict))
    for mode in unknown:
        problems.append(f"unknown failure_mode '{mode}' - must be one of {list(FAILURE_MODES)}")

    if problems:
        print("FATAL: golden_set.yaml is malformed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(2)
    return cases


DEFAULT_THRESHOLDS = {
    "min_cases": 30,
    "min_failure_modes": 5,
    "faithfulness_floor": 0.85,
    "precision_floor": 0.80,
    "recall_floor": 0.75,
    "max_p95_latency_ms": 4000,
    "max_cost_per_case_usd": 0.02,
}


def load_thresholds() -> dict:
    data = load_yaml(THRESHOLDS_PATH)
    merged = dict(DEFAULT_THRESHOLDS)
    for key, default in DEFAULT_THRESHOLDS.items():
        if data.get(key) is not None:
            merged[key] = data[key]
        else:
            print(f"[warn] thresholds.yaml has no '{key}', using default {default}")
    per_mode = data.get("per_mode_recall_floor") or {}
    merged["per_mode_recall_floor"] = per_mode if isinstance(per_mode, dict) else {}
    return merged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Small n here, so no interpolation games."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def run_cases(cases: list[dict], judge, system) -> list[dict]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            actual = system(case["input"])
        except Exception as exc:  # a broken system is a score of zero, not a crash
            actual = ""
            print(f"[warn] system raised on {case['id']}: {type(exc).__name__}: {exc}")
        latency_ms = (time.perf_counter() - started) * 1000.0
        verdict = judge.score(case, actual)
        rows.append(
            {
                "id": case["id"],
                "failure_mode": case["failure_mode"],
                "faithfulness": verdict.faithfulness,
                "mode_correct": verdict.mode_correct,
                "predicted_mode": verdict.predicted_mode,
                "rationale": verdict.rationale,
                "latency_ms": latency_ms,
                # Cost is 0.0 until you instrument it in src/observability/.
                # A cost gate you never populate is a gate that always passes,
                # which is worse than no gate because it looks like coverage.
                "cost_usd": 0.0,
            }
        )
    return rows


def per_mode_stats(rows: list[dict]) -> dict:
    """Precision and recall per failure mode, one-vs-rest."""
    stats = {}
    for mode in FAILURE_MODES:
        actual_pos = [r for r in rows if r["failure_mode"] == mode]
        pred_pos = [r for r in rows if r["predicted_mode"] == mode]
        tp = len([r for r in pred_pos if r["failure_mode"] == mode])
        precision = tp / len(pred_pos) if pred_pos else 0.0
        recall = tp / len(actual_pos) if actual_pos else 0.0
        faith = (
            statistics.fmean([r["faithfulness"] for r in actual_pos])
            if actual_pos
            else 0.0
        )
        stats[mode] = {
            "n": len(actual_pos),
            "precision": precision,
            "recall": recall,
            "faithfulness": faith,
            "present": bool(actual_pos),
        }
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

BAR = "=" * 78


def print_table(stats: dict, thresholds: dict) -> None:
    per_mode_floor = thresholds.get("per_mode_recall_floor") or {}
    print(f"\n{'failure mode':<20}{'n':>4}{'precision':>12}{'recall':>10}"
          f"{'faithful':>11}{'recall floor':>15}")
    print("-" * 78)
    for mode, s in stats.items():
        floor = per_mode_floor.get(mode, thresholds["recall_floor"])
        if not s["present"]:
            print(f"{mode:<20}{0:>4}{'MODE ABSENT FROM GOLDEN SET':>48}")
            continue
        print(
            f"{mode:<20}{s['n']:>4}{s['precision']:>12.2f}{s['recall']:>10.2f}"
            f"{s['faithfulness']:>11.2f}{floor:>15.2f}"
        )
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Northwind Freight evaluation gate")
    parser.add_argument("--llm", action="store_true", help="try the LLM judge")
    parser.add_argument("--json", action="store_true", help="also print JSON summary")
    args = parser.parse_args()

    cases = load_cases()
    thresholds = load_thresholds()
    judge = get_judge(prefer_llm=args.llm)
    system, system_name = load_system()

    print(BAR)
    print("NORTHWIND FREIGHT - EVALUATION GATE")
    print(BAR)
    print(f"golden set : {GOLDEN_SET_PATH.name} ({len(cases)} cases)")
    print(f"judge      : {judge.name}")
    print(f"system     : {system_name}")

    rows = run_cases(cases, judge, system)
    stats = per_mode_stats(rows)

    modes_present = sorted({r["failure_mode"] for r in rows})
    overall_faith = statistics.fmean([r["faithfulness"] for r in rows]) if rows else 0.0
    matched = [r for r in rows if r["predicted_mode"]]
    overall_precision = (
        len([r for r in matched if r["mode_correct"]]) / len(matched) if matched else 0.0
    )
    overall_recall = (
        len([r for r in rows if r["mode_correct"]]) / len(rows) if rows else 0.0
    )
    p95_latency = percentile([r["latency_ms"] for r in rows], 95)
    mean_cost = statistics.fmean([r["cost_usd"] for r in rows]) if rows else 0.0

    print_table(stats, thresholds)

    failures: list[str] = []

    # Coverage gates first. They are the reason a fresh clone goes red, and they
    # are listed first because no quality number is trustworthy under them.
    if len(cases) < thresholds["min_cases"]:
        failures.append(
            f"COVERAGE: golden set has {len(cases)} cases, gate requires "
            f"{thresholds['min_cases']}. Short by "
            f"{thresholds['min_cases'] - len(cases)}. Build them from "
            f"customer/tickets.jsonl."
        )
    if len(modes_present) < thresholds["min_failure_modes"]:
        missing = [m for m in FAILURE_MODES if m not in modes_present]
        failures.append(
            f"COVERAGE: golden set covers {len(modes_present)} of "
            f"{thresholds['min_failure_modes']} required failure modes. "
            f"Missing: {', '.join(missing)}."
        )

    if overall_faith < thresholds["faithfulness_floor"]:
        failures.append(
            f"QUALITY: faithfulness {overall_faith:.2f} is below floor "
            f"{thresholds['faithfulness_floor']:.2f}."
        )
    if overall_precision < thresholds["precision_floor"]:
        failures.append(
            f"QUALITY: precision {overall_precision:.2f} is below floor "
            f"{thresholds['precision_floor']:.2f}."
        )

    per_mode_floor = thresholds.get("per_mode_recall_floor") or {}
    if per_mode_floor:
        for mode, floor in per_mode_floor.items():
            s = stats.get(mode)
            if s and s["present"] and s["recall"] < float(floor):
                failures.append(
                    f"QUALITY: recall on {mode} is {s['recall']:.2f}, below its "
                    f"asymmetric floor {float(floor):.2f}."
                )
    if overall_recall < thresholds["recall_floor"]:
        failures.append(
            f"QUALITY: recall {overall_recall:.2f} is below floor "
            f"{thresholds['recall_floor']:.2f}."
        )

    if p95_latency > thresholds["max_p95_latency_ms"]:
        failures.append(
            f"OPERATIONAL: p95 latency {p95_latency:.0f}ms exceeds "
            f"{thresholds['max_p95_latency_ms']}ms."
        )
    if mean_cost > thresholds["max_cost_per_case_usd"]:
        failures.append(
            f"OPERATIONAL: cost/case ${mean_cost:.4f} exceeds "
            f"${thresholds['max_cost_per_case_usd']:.4f}."
        )

    print(f"\noverall    faithfulness {overall_faith:.2f}   "
          f"precision {overall_precision:.2f}   recall {overall_recall:.2f}")
    print(f"operational  p95 latency {p95_latency:.0f}ms   "
          f"cost/case ${mean_cost:.4f}")

    if args.json:
        print("\nJSON_SUMMARY " + json.dumps(
            {
                "cases": len(cases),
                "modes_present": modes_present,
                "faithfulness": round(overall_faith, 4),
                "precision": round(overall_precision, 4),
                "recall": round(overall_recall, 4),
                "p95_latency_ms": round(p95_latency, 2),
                "cost_per_case_usd": round(mean_cost, 6),
                "failures": failures,
            },
            sort_keys=True,
        ))

    print(BAR)
    if failures:
        print("GATE FAILED")
        for f in failures:
            print(f"  FAIL  {f}")
        print(BAR)
        print(
            "This is the expected result on a fresh clone. Work the COVERAGE "
            "lines first:\n"
            "  a quality score computed over 12 cases is not a measurement, it "
            "is an anecdote.\n"
            "Run again after every batch of cases you add. Watching this number "
            "move is the job."
        )
        return 1

    print("GATE PASSED")
    print(BAR)
    print(
        "Before you put this in a deck: can you defend every threshold in "
        "eval/thresholds.yaml\nin one sentence each? If not, the gate passed "
        "and you still cannot ship it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""CLI for all four phases.

    python run.py benchmark                      Phase I   - what is in the benchmark
    python run.py eval --criterion specific      Phase I   - score a rubric on a split
    python run.py tune --criterion specific      Phase II  - run RART, stage the result
    python run.py ablation --criterion specific  Phase II  - RART vs vanilla
    python run.py promote --criterion specific   Phase II  - staged rubric goes live
    python run.py serve --record rec-001         Phase III - generate/judge/revise
    python run.py curve                          Phase III - pass rate vs retry budget
    python run.py monitor --week 6               Phase IV  - the drift band
    python run.py augment --week 6               Phase IV  - grow the benchmark

Add ``--offline`` to any of them to run every role on the deterministic rule
engine: no API key, no network, no cost. That is the recommended first run -
walk all four phases, see what each one does, and only then decide whether to
spend anything. The default without the flag is Gemini, configured per role in
configs/base.yaml.

The CLI goes through ``src/services/`` for the same reason the API does: a
guardrail the CLI can skip is a guardrail. There is one orchestration path, not
two.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from src.services import (
    BenchmarkService,
    MonitoringService,
    ServingService,
    TuningService,
)
from src.services._context import build_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="LLM judge lifecycle: four phases, one CLI."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="print the raw result and nothing else"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run every role on the stub provider: no API key, no network, no "
        "cost. Numbers produced this way measure the rubric's rules, not a "
        "model's judgement.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("benchmark", help="Phase I: benchmark composition and splits")

    p = sub.add_parser("eval", help="Phase I: score a rubric on one split")
    p.add_argument("--criterion", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--staged", action="store_true", help="score the staged rubric")

    p = sub.add_parser("tune", help="Phase II: run RART and stage the result")
    p.add_argument("--criterion", required=True)
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="vanilla ablation: label mismatches only, no reasoning meta-judge",
    )

    p = sub.add_parser("ablation", help="Phase II: RART vs vanilla on the same splits")
    p.add_argument("--criterion", required=True)

    p = sub.add_parser("promote", help="Phase II: promote a staged rubric to live")
    p.add_argument("--criterion", required=True)

    p = sub.add_parser("serve", help="Phase III: generate, judge, revise")
    p.add_argument("--record", help="record id; omit to serve the whole catalogue")
    p.add_argument("--max-retries", type=int)

    p = sub.add_parser("curve", help="Phase III: pass rate vs revision budget")
    p.add_argument("--max-k", type=int, default=6)

    p = sub.add_parser("monitor", help="Phase IV: check the drift band for a week")
    p.add_argument("--week", type=int, required=True)

    p = sub.add_parser("augment", help="Phase IV: turn a rated week into benchmark rows")
    p.add_argument("--week", type=int, required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # A missing key, an unknown criterion or a missing week are configuration
    # problems, not crashes. A traceback buries the one line that says how to fix
    # it under nine that do not, and every message raised in this project is
    # written to be the thing the reader acts on.
    try:
        context = build_context(offline=args.offline)
        if not args.json:
            _banner(context)
        result = _dispatch(args, context)
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
        # KeyError stringifies to its repr, so a carefully written message comes
        # out wrapped in quotes. Strip them.
        message = str(exc)[1:-1] if isinstance(exc, KeyError) else str(exc)
        print(f"\n{message}\n", file=sys.stderr)
        return 2

    if args.json:
        # Provenance is attached HERE rather than at each service, so the claim
        # "every number carries what produced it" is true of the CLI as well as
        # the API. It was not, briefly: --json emitted the bare service result,
        # so a number could be piped into a report with no record of which model
        # made it or whether the run was the offline rule engine.
        print(
            json.dumps(
                {**result, "provenance": context.runtime.provenance()},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _render(args.command, result, context)
    return 0


def _dispatch(args: argparse.Namespace, context: Any) -> dict[str, Any]:
    if args.command == "benchmark":
        return BenchmarkService(context).report()
    if args.command == "eval":
        return BenchmarkService(context).evaluate(
            args.criterion, split=args.split, staged=args.staged
        )
    if args.command == "tune":
        return TuningService(context).tune(
            args.criterion,
            reasoning_alignment=not args.no_reasoning,
        )
    if args.command == "ablation":
        return TuningService(context).ablation(args.criterion)
    if args.command == "promote":
        return TuningService(context).promote(args.criterion)
    if args.command == "serve":
        service = ServingService(context)
        if args.record:
            return service.serve(args.record, max_retries=args.max_retries)
        return service.serve_all()
    if args.command == "curve":
        return ServingService(context).retry_curve(max_k=args.max_k)
    if args.command == "monitor":
        return MonitoringService(context).check(args.week)
    if args.command == "augment":
        return MonitoringService(context).augment(args.week)
    raise SystemExit(f"unknown command {args.command!r}")


def _banner(context: Any) -> None:
    """Say what produced these numbers, before producing them.

    Two warnings, and they are printed rather than logged because the failure
    mode is a person copying a number into a slide, not a program misbehaving.
    """
    config = context.config
    roles = ", ".join(
        f"{role}={config.role(role).provider}/{config.role(role).model or 'stub'}"
        for role in config.roles
    )
    print(f"domain={config.domain_name}  {roles}")

    if config.is_fully_offline:
        print(
            "\n  OFFLINE RUN (provider: stub for every role).\n"
            "  These numbers measure the RULES in the rubric, not any model's\n"
            "  judgement. They are a baseline to beat, not a model result.\n"
        )
    elif config.single_model_config:
        print(
            "\n  The generator and judge are the same model. Self-preference\n"
            "  bias is uncontrolled: the judge may be rewarding its own house\n"
            "  style rather than quality. Fine for learning; state it if you\n"
            "  report the numbers.\n"
        )


def _render(command: str, result: dict[str, Any], context: Any) -> None:
    if command == "benchmark":
        print("\nBenchmark composition\n" + "-" * 62)
        for cid, row in result["splits"]["criteria"].items():
            gate = "gate" if row["must_have"] else "soft"
            flag = "" if row["balanced"] else "   <- not balanced"
            print(
                f"  {cid:<10} {gate:<5} n={row['n']:<4} "
                f"fail={row['fail_fraction']:<6} splits={row['splits']}{flag}"
            )
        print(
            "\n  Class-balanced and difficulty-enriched on purpose. These are "
            "alignment\n  metrics, NOT live defect rates. See "
            "domains/*/labels.jsonl for why.\n"
        )
        return

    if command in ("eval",):
        m = result["metrics"]
        print(f"\n{result['criterion']} on {result['split']} (n={result['n']})\n" + "-" * 62)
        for key in ("specificity", "recall", "reasoning_agreement"):
            lo, hi = m["ci95"][key]
            print(f"  {key:<22} {_fmt(m[key])}   95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  {'weighted':<22} {m['weighted']:.3f}")
        print(f"\n  confusion: {result['confusion']}")
        if result["false_passes"]:
            print(f"  false passes (would reach users): {result['false_passes']}")
        if result["false_fails"]:
            print(f"  false fails  (good work rejected): {result['false_fails']}")
        print(
            "\n  The intervals are wide because the benchmark is small. Treat a\n"
            "  gap narrower than the intervals as noise.\n"
        )
        return

    if command == "tune":
        print(f"\nRART: {result['criterion']}   (reasoning_alignment="
              f"{result['reasoning_alignment']})\n" + "-" * 62)
        print("  iter  weighted  spec    rec     ra      focus  kept")
        for it in result["iterations"]:
            v = it["validation"]
            print(
                f"  {it['index']:<5} {it['weighted']:<9.3f} "
                f"{_fmt(v['specificity']):<7} {_fmt(v['recall']):<7} "
                f"{_fmt(v['reasoning_agreement']):<7} {it['focus_size']:<6} "
                f"{'*' if it['accepted'] else ''}"
            )
        print(f"\n  best iteration : {result['best_iteration']}")
        print(f"  stopped        : {result['stopped_because']}")
        print(f"  improved on seed: {result['improved_on_seed']}")
        if result.get("staged_at"):
            print(f"  staged         : {result['staged_at']}")
            print(f"  promote with   : {result['promote_with']}")
        if not result["improved_on_seed"]:
            print("\n" + _null_result_note(result))
        return

    if command == "curve":
        print("\nCumulative pass rate vs revision budget k\n" + "-" * 62)
        for point in result["curve"]:
            bar = "#" * round(point["cumulative_pass_rate"] * 40)
            print(
                f"  k={point['k']:<3} {point['cumulative_pass_rate']:.3f}  "
                f"{bar} ({point['passed']}/{point['total']})"
            )
        print(f"\n  {result['reading_guide']}\n")
        return

    if command == "monitor":
        print(f"\nDrift check, week {result['week']}  "
              f"(n={result['n_rated']}, new items={result['n_new_items']})\n" + "-" * 62)
        for report in result["reports"]:
            print(f"\n  criterion: {report['criterion']}")
            for scope, checks in (("overall", report["overall"]), ("new items", report["new_items"])):
                for check in checks:
                    state = "in band" if check["in_band"] else "OUT OF BAND"
                    print(
                        f"    {scope:<10} {check['metric']:<12} "
                        f"judge={_fmt(check['judge'])} "
                        f"raters={check['rater_mean']:.3f}+-{check['rater_sd']:.3f} "
                        f"floor={check['lower_bound']:.3f}  n={check['n']}  {state}"
                    )
            for note in report["notes"]:
                print(f"\n    NOTE: {note}")
        print(f"\n  alert : {result['alert']}")
        print(f"  action: {result['action'].get('note')}")
        for command_line in result["action"].get("commands", []):
            print(f"          {command_line}")
        print()
        return

    print(json.dumps(result, indent=2, ensure_ascii=False))


def _null_result_note(result: dict[str, Any]) -> str:
    """Explain a null result WITHOUT flattering it.

    "No improvement" has two completely different meanings and reporting them
    with the same sentence is how a broken optimiser gets read as a validated
    one. If the seed is already scoring 0.95, there was nothing to find. If it
    is scoring 0.00, there was plenty to find and the optimiser did not find it,
    which is a finding about the OPTIMISER and needs saying in those words.
    """
    best = next(
        (it for it in result["iterations"] if it["index"] == result["best_iteration"]),
        None,
    )
    spec = (best or {}).get("validation", {}).get("specificity")

    if spec is not None and spec >= 0.85:
        return (
            "  No improvement over the seed rubric, and specificity is already\n"
            f"  {spec:.3f}. The human guideline was at ceiling for this criterion\n"
            "  and there was nothing for the optimiser to find. That is a RESULT.\n"
            "  Report it; do not quietly drop the criterion from the table.\n"
        )

    return (
        "  No improvement over the seed rubric, and specificity is only\n"
        f"  {_fmt(spec)}. This is NOT a criterion at ceiling - there is plenty of\n"
        "  headroom and this optimiser could not reach it.\n\n"
        "  Offline, that is expected and instructive: the stub reflector can only\n"
        "  learn lexical rules (ban a phrase, require a token), and these failures\n"
        "  are not lexical. Spotting a spoiler phrased in words nobody listed, or\n"
        "  a number attached to the wrong noun, requires reading the artefact\n"
        "  against the record. That is the gap a model judge is FOR, and it is\n"
        "  the number to beat when you set a provider and run this again.\n"
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())

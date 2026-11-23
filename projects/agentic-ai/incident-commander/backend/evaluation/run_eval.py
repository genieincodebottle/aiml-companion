"""Evaluation harness for the Incident Commander.

    uv run python -m evaluation.run_eval                  # live Gemini, 1 repeat (a smoke pass)
    uv run python -m evaluation.run_eval --repeats 3      # what `make eval` runs; real intervals
    uv run python -m evaluation.run_eval --judge          # + LLM-as-judge scoring
    uv run python -m evaluation.run_eval --stub           # deterministic, free, CI
    uv run python -m evaluation.run_eval --check-baseline # fail on regression
    uv run python -m evaluation.run_eval --engine graph   # evaluate the graph engine

## What this measures, and why it is shaped this way

Runs every scenario in the fault catalog N times with an approve-all operator, then
reports four families of metric: outcome, efficiency, safety, and process quality
(see `metrics.py` for the reasoning behind each).

Three choices are worth calling out, because they are the difference between an
eval that tells you something and one that flatters you:

1. **N repeats, not one.** The system is stochastic. One run per scenario is an
   anecdote, and the `stability` column - how often repeats agreed - is usually
   more informative than the headline rate. (The CLI default is still 1, priced
   as a smoke pass; `--repeats` and the Makefile's `eval` target are where the
   real measurement lives. With one repeat, stability is vacuously 100%.)

2. **Negative scenarios count.** `unfixable_ca_expiry` must ESCALATE and
   `false_alarm` must do nothing. Scoring against `expected_outcome` rather than
   "resolved" is the only way an agent can be penalized for confidently fixing
   something that was never broken.

3. **`--stub` runs the identical harness deterministically.** CI therefore checks
   that the eval itself works, and that the loop's control flow has not regressed,
   on every push and for free. Stub numbers are not agent quality - they are
   harness health - and the report says so rather than letting anyone confuse them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python evaluation/run_eval.py` as well as `python -m evaluation.run_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.judge import judge_run, judgement_to_dict  # noqa: E402
from evaluation.metrics import aggregate, compare_to_baseline, is_correct_outcome, score_trajectory  # noqa: E402
from src.agents import PROMPT_VERSION  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm import token_tracker  # noqa: E402
from src.loop import IncidentManager  # noqa: E402
from src.models.state import EventType, IncidentStatus  # noqa: E402
from src.sandbox.scenarios import SCENARIOS, list_scenarios  # noqa: E402
from src.tools.observability import observe_all  # noqa: E402

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _build_llm(stub: bool):
    if not stub:
        return None  # the manager resolves the real Gemini client
    from src.offline_llm import OfflineLLM

    return OfflineLLM()  # no per-call delay: nobody is watching a console here


async def run_once(scenario_id: str, *, stub: bool, repeat: int) -> dict[str, Any]:
    """Run one incident to completion with an approve-all operator."""
    manager = IncidentManager(llm=_build_llm(stub))
    scenario = SCENARIOS[scenario_id]
    started = time.time()

    state = manager.trigger(scenario_id, None, auto_approve=False)
    # Snapshot the telemetry the agent will see, so the judge can check its citations.
    telemetry = observe_all(manager.stacks[state.incident_id])

    routes: list[str] = []
    async for ev in manager.subscribe(state.incident_id):
        if ev.type == EventType.ROUTE:
            routes.append(ev.payload.get("route", ""))
        if ev.type == EventType.AWAITING_APPROVAL:
            # The harness is the approver here; the audit trail says so rather than "anonymous".
            manager.resolve_approval(
                state.incident_id, ev.payload["action_id"], "approve", "eval", actor="eval-harness"
            )

    s = manager.get(state.incident_id)
    outcome = {
        IncidentStatus.RESOLVED: "resolved",
        IncidentStatus.ESCALATED: "escalated",
        IncidentStatus.CANCELLED: "cancelled",
    }.get(s.status, "failed")

    record: dict[str, Any] = {
        "scenario": scenario_id,
        "repeat": repeat,
        "incident_id": s.incident_id,
        "service": s.service,
        "engine": s.engine,
        "outcome": outcome,
        "correct_outcome": is_correct_outcome(outcome, s.halt_reason.value if s.halt_reason else None, scenario),
        "halt_reason": s.halt_reason.value if s.halt_reason else None,
        "iterations": s.iterations,
        "evidence_rounds": s.evidence_rounds,
        "cost_usd": round(s.total_cost_usd, 6),
        "tokens": s.total_tokens,
        "duration_s": round(time.time() - started, 2),
        "routes": routes,
        "telemetry": telemetry,
        "triage": s.triage_summary,
        "diagnoses": s.diagnostics,
        "evidence": s.evidence,
        "actions": [a.model_dump(mode="json") for a in s.actions],
        "critiques": [a.critique for a in s.actions if a.critique],
        "verifications": s.verification_history,
        "guardrail_flags": s.guardrail_flags,
        "postmortem": s.postmortem,
        "by_agent": s.agent_usage,
    }
    record["trajectory"] = score_trajectory(record, scenario)
    # The token ledger is process-global and normally pruned by the manager's
    # retention cap - but this harness builds a fresh manager per run, so no
    # manager ever hits its cap and a long --repeats sweep would accumulate every
    # run's per-agent usage forever. The figures are already in the record.
    token_tracker.forget(s.incident_id)
    return record


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not args.stub and not settings.llm_enabled:
        print(
            "GOOGLE_API_KEY is not configured.\n"
            "Run the live evaluation with a key, or use --stub for the deterministic "
            "harness check (which measures the harness, not the agent).",
            file=sys.stderr,
        )
        return 2

    scenario_ids = args.scenarios or [s["id"] for s in list_scenarios()]
    unknown = [s for s in scenario_ids if s not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    total_runs = len(scenario_ids) * args.repeats
    mode = "STUB (deterministic)" if args.stub else f"LIVE ({settings.model_name})"
    print(f"\n=== Incident Commander Evaluation - {mode} ===")
    print(f"{len(scenario_ids)} scenario(s) x {args.repeats} repeat(s) = {total_runs} run(s), engine={settings.engine}")
    print(f"prompt_version={PROMPT_VERSION}\n")

    runs: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for scenario_id in scenario_ids:
            record = await run_once(scenario_id, stub=args.stub, repeat=repeat)
            runs.append(record)
            mark = "OK " if record["correct_outcome"] else "BAD"
            print(
                f"  [{mark}] {scenario_id:<22} run {repeat + 1}/{args.repeats}  "
                f"{record['outcome']:<10} iters={record['iterations']} "
                f"traj={record['trajectory']['score']:.2f} ${record['cost_usd']:.5f}"
            )

    # --- LLM-as-judge (optional, costs real quota) --------------------------
    if args.judge:
        if args.stub:
            print("\n--judge requires a live model; skipping (stub mode has nothing to judge).")
        else:
            from src.llm import get_llm

            llm = get_llm()
            print(f"\nJudging {len(runs)} run(s)...")
            for record in runs:
                try:
                    record["judge"] = judgement_to_dict(judge_run(record, llm))
                except Exception as exc:  # noqa: BLE001 - a judge failure must not lose run data
                    print(f"  judge failed for {record['scenario']}: {exc}", file=sys.stderr)
                    record["judge"] = None

    report = aggregate(runs, SCENARIOS)
    report["meta"] = {
        "mode": "stub" if args.stub else "live",
        "model": "stub" if args.stub else settings.model_name,
        "engine": settings.engine,
        "prompt_version": PROMPT_VERSION,
        "repeats": args.repeats,
        "judged": bool(args.judge and not args.stub),
        "generated_at": time.time(),
    }

    _print_report(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"report": report, "runs": runs}, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results written to {out}")

    exit_code = 0
    if args.check_baseline:
        exit_code = _check_baseline(report)
    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nBaseline saved to {BASELINE_PATH}")
    return exit_code


def _print_report(report: dict[str, Any]) -> None:
    o = report["overall"]
    print("\n" + "=" * 108)
    print(
        f"{'scenario':<22}{'expect':<10}{'correct':<9}{'stab':<7}{'iters':<7}"
        f"{'first':<7}{'traj':<7}{'harm':<7}{'inject':<8}{'judge':<7}{'cost':<9}"
    )
    print("-" * 108)
    for row in report["by_scenario"]:
        inject = "-" if row["injection_resisted_rate"] is None else f"{row['injection_resisted_rate']:.0%}"
        judge = "-" if row["judge_overall"] is None else f"{row['judge_overall']:.1f}"
        print(
            f"{row['scenario']:<22}{row['expected_outcome']:<10}"
            f"{row['correct_outcome_rate']:<9.0%}{row['stability']:<7.0%}"
            f"{row['mean_iterations']:<7.1f}{row['correct_first_action_rate']:<7.0%}"
            f"{row['trajectory_score']:<7.2f}{row['harmful_action_rate']:<7.0%}"
            f"{inject:<8}{judge:<7}${row['mean_cost_usd']:<8.5f}"
        )
    print("-" * 108)
    ci = o["correct_outcome_ci95"]
    print("\nOUTCOME")
    print(f"  Correct outcome        : {o['correct_outcome_rate']:.1%}  (95% CI {ci[0]:.1%}-{ci[1]:.1%})")
    print(f"  Resolution (positives) : {o['resolution_rate_positive_only']:.1%}")
    print(f"  Escalation rate        : {o['escalation_rate']:.1%}   <- correct on negative scenarios")
    print("\nEFFICIENCY")
    print(f"  Correct first action   : {o['correct_first_action_rate']:.1%}")
    print(f"  Mean iterations (MTTR) : {o['mean_iterations']:.2f}")
    print(f"  Mean cost / incident   : ${o['mean_cost_usd']:.5f}   (total ${o['total_cost_usd']:.4f})")
    print(f"  Mean duration          : {o['mean_duration_s']:.1f}s")
    print("\nSAFETY")
    print(f"  Harmful action rate    : {o['harmful_action_rate']:.1%}   <- lower is better; 0% is the bar")
    print("\nPROCESS")
    print(f"  Trajectory score       : {o['trajectory_score']:.2f}   (deterministic checks, see metrics.py)")
    if o["judge_overall"] is not None:
        print(f"  Judge overall          : {o['judge_overall']:.2f}/5")
        print(f"  Judge evidence grounding: {o['judge_evidence_grounding']:.2f}/5")
        print(f"  Judge hallucination-free: {o['judge_hallucination_free']:.2f}/5")
        print(f"  Trusted unattended     : {o['trusted_unattended_rate']:.1%}")
    print("\nRELIABILITY")
    print(f"  Stability (repeat agreement): {o['stability']:.1%}")
    if o["flaky_scenarios"]:
        print(f"  Flaky scenarios        : {', '.join(o['flaky_scenarios'])}   <- non-deterministic outcome")
    else:
        print("  Flaky scenarios        : none")
    if report["meta"]["mode"] == "stub":
        print(
            "\nNOTE: stub mode measures the HARNESS and the loop's control flow, not agent\n"
            "      quality. Run without --stub for a real measurement of the model."
        )
    print()


def _check_baseline(report: dict[str, Any]) -> int:
    if not BASELINE_PATH.exists():
        print(f"\nNo baseline at {BASELINE_PATH}; run with --save-baseline first.")
        return 0
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    diff = compare_to_baseline(report["overall"], baseline.get("overall", {}))
    print("\nBASELINE COMPARISON")
    for entry in diff["improvements"]:
        print(f"  improved  {entry['metric']}: {entry['baseline']} -> {entry['current']} ({entry['delta']:+.4f})")
    for entry in diff["regressions"]:
        print(f"  REGRESSED {entry['metric']}: {entry['baseline']} -> {entry['current']} ({entry['delta']:+.4f})")
    if diff["passed"]:
        print("  no regressions beyond tolerance")
        return 0
    print("\nFAILED: metrics regressed against the committed baseline.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the Incident Commander agent loop.")
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "runs per scenario. Default 1 because a live 10-scenario pass costs about "
            "$0.25 and 12 minutes; raise it for tighter confidence intervals, and note "
            "that --stub is free and instant"
        ),
    )
    p.add_argument("--scenarios", nargs="*", help="subset of scenario ids (default: all)")
    p.add_argument("--stub", action="store_true", help="deterministic stub LLM: free, offline, for CI")
    p.add_argument("--judge", action="store_true", help="also score reasoning with an LLM judge (costs quota)")
    p.add_argument("--out", help="write the full JSON report and per-run records here")
    p.add_argument("--save-baseline", action="store_true", help="write this report to evaluation/baseline.json")
    p.add_argument("--check-baseline", action="store_true", help="exit non-zero if metrics regressed")
    p.add_argument("--engine", choices=["loop", "graph"], help="override IC_ENGINE for this run")
    p.add_argument(
        "--persist-memory",
        action="store_true",
        help="read and write the real runbook memory file (default: a temporary, empty one)",
    )
    return p


def main() -> int:
    import os
    import tempfile

    from src.config import reset_settings_cache
    from src.memory.runbook import reset_memory_cache

    args = build_parser().parse_args()
    if args.engine:
        os.environ["IC_ENGINE"] = args.engine
    if not args.persist_memory:
        # Each eval starts from an empty memory. Otherwise every run writes episodes
        # into backend/data/runbook_memory.json, later runs recall them, and the
        # numbers drift from one eval to the next for reasons unrelated to the agent.
        os.environ["IC_MEMORY_PATH"] = str(Path(tempfile.mkdtemp(prefix="ic-eval-")) / "memory.json")
    reset_settings_cache()
    reset_memory_cache()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

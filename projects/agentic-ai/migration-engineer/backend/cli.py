"""Command-line entrypoint for the Autonomous Migration Engineer.

    python cli.py list                       # show the fleet job catalog
    python cli.py run datetime-fleet         # run a job through the LangGraph batch
    python cli.py run datetime-fleet --policy dry_run
    python cli.py stream datetime-fleet      # run the LIVE engine, printing SSE events

`run` uses the declarative LangGraph pipeline (batch, non-interactive). `stream` uses
the live async engine with auto-approve so you can watch the fan-out and the agent loop
in the terminal. Both work offline on the deterministic worker; set ANTHROPIC_API_KEY
(and `uv sync --extra sdk`) to drive the real Claude Agent SDK worker.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.config import get_settings
from src.models.state import EventType
from src.orchestrator.engine import MigrationManager
from src.orchestrator.graph import run_batch
from src.rulebook.rules import get_job, list_jobs


def _cmd_list() -> None:
    print(f"\nExecution mode: {get_settings().execution_mode}\n")
    print(f"{'job id':<18}{'rule':<34}{'repos':<7}title")
    print("-" * 90)
    for j in list_jobs():
        print(f"{j['id']:<18}{j['rule_name'][:32]:<34}{j['repo_count']:<7}{j['title']}")
    print()


async def _cmd_run(job_id: str, policy: str) -> None:
    job = get_job(job_id)
    if job is None:
        raise SystemExit(f"unknown job '{job_id}' (try: python cli.py list)")
    print(f"\nRunning '{job.title}' via LangGraph batch  (mode={get_settings().execution_mode}, policy={policy})\n")
    summary = await run_batch(job, approval_policy=policy)

    print(f"{'repo':<26}{'outcome':<20}{'tests':<8}{'files':<7}{'steps':<6}tampered")
    print("-" * 80)
    for r in summary["repos"]:
        print(
            f"{r['name']:<26}{str(r['outcome']):<20}{str(r['tests_passing']):<8}"
            f"{len(r['files_changed']):<7}{r['steps']:<6}{r['tampered_with_tests']}"
        )
    print("-" * 80)
    print(f"By outcome     : {summary['by_outcome']}")
    print(f"Total cost USD : {summary['total_cost_usd']}\n")


async def _cmd_stream(job_id: str) -> None:
    if get_job(job_id) is None:
        raise SystemExit(f"unknown job '{job_id}' (try: python cli.py list)")
    manager = MigrationManager()
    state = manager.trigger(job_id, auto_approve=True)  # auto-approve so the demo flows
    print(f"\nLive run {state.job_id}  (mode={state.mode})\n")
    async for ev in manager.subscribe(state.job_id):
        tag = f"[{ev.repo_id or 'job'}]"
        text = ev.payload.get("text") or ev.payload.get("summary") or ev.payload.get("reason") or ""
        extra = ""
        if ev.type == EventType.TOOL_CALL:
            extra = f"{ev.payload.get('tool')} -> {ev.payload.get('summary')}"
        elif ev.type == EventType.TESTS_RUN:
            extra = "PASS" if ev.payload.get("passed") else "FAIL"
        elif ev.type == EventType.GUARDRAIL_BLOCK:
            extra = f"BLOCKED {ev.payload.get('tool')}: {ev.payload.get('reason')}"
        print(f"{tag:<24}{ev.type.value:<18}{text or extra}")
    print()


def _utf8_output() -> None:
    """Make printing model text safe when output is piped or redirected.

    On Windows a redirected stdout uses the ANSI code page (cp1252), which cannot
    encode characters a live model often writes, such as an arrow. The stub worker
    only prints ASCII, so without this the failure appears only in live mode, as a
    UnicodeEncodeError partway through `stream ... | grep ...`.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _utf8_output()
    parser = argparse.ArgumentParser(prog="migrate", description="Autonomous Migration Engineer CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the fleet job catalog")
    p_run = sub.add_parser("run", help="run a job through the LangGraph batch pipeline")
    p_run.add_argument("job_id")
    p_run.add_argument("--policy", choices=("auto", "dry_run"), default="auto")
    p_stream = sub.add_parser("stream", help="run the live engine and print SSE events")
    p_stream.add_argument("job_id")

    args = parser.parse_args()
    if args.cmd == "list":
        _cmd_list()
    elif args.cmd == "run":
        asyncio.run(_cmd_run(args.job_id, args.policy))
    elif args.cmd == "stream":
        asyncio.run(_cmd_stream(args.job_id))


if __name__ == "__main__":
    main()

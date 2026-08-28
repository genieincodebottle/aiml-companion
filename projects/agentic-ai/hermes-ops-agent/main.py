#!/usr/bin/env python3
"""hermes-ops-agent: measure whether the Hermes skill loop actually pays off.

Four commands:

    sessions    list sessions from a Hermes state.db, so you can pick two ids
    compare     compare two sessions that ran the same task
    demo        run compare against the recorded fixture (no Hermes needed)
    probe       write prompt-injection payloads, or scan a session for the canary

Everything reads the database read-only. Nothing here writes to ~/.hermes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.state_db import StateDb, StateDbError, default_db_path  # noqa: E402
from src.metrics import compare  # noqa: E402
from src import report, injection, fixture  # noqa: E402

FIXTURE_DB = Path(__file__).parent / "fixtures" / "state.fixture.db"


def _open(db_arg: str | None) -> StateDb:
    return StateDb.open(db_arg) if db_arg else StateDb.open()


def cmd_sessions(args: argparse.Namespace) -> int:
    with _open(args.db) as db:
        rows = db.sessions(source=args.source, limit=args.limit)
        print(f"# {db.path}   schema v{db.schema_version() or '?'}\n")
        print(report.render_session_table(rows))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with _open(args.db) as db:
        base = db.session(args.baseline)
        cand = db.session(args.candidate)
        if base is None:
            print(f"No session {args.baseline!r} in {db.path}", file=sys.stderr)
            return 2
        if cand is None:
            print(f"No session {args.candidate!r} in {db.path}", file=sys.stderr)
            return 2

        result = compare(
            base,
            cand,
            baseline_tools=db.tool_names(base.id),
            candidate_tools=db.tool_names(cand.id),
            baseline_recount=db.counted_tool_calls(base.id),
            candidate_recount=db.counted_tool_calls(cand.id),
        )

    if args.json:
        out = report.render_json(result)
    else:
        out = report.render_text(result)
    print(out)

    if args.out:
        Path(args.out).write_text(report.render_json(result), encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)

    # Exit non-zero when the skill did not shorten anything, so this is usable
    # as a check in CI once you have a stable task to run against.
    tool_delta = next(d for d in result.deltas if d.name == "tool_calls")
    return 0 if tool_delta.change < 0 else 1


def cmd_demo(args: argparse.Namespace) -> int:
    # Build the fixture if this is a fresh clone. It is deterministic and takes
    # milliseconds, so making the learner run a separate command first - one
    # that needs `make`, which Windows usually lacks - would turn the very
    # first command in the README into a dead end.
    try:
        fixture.ensure()
    except fixture.FixtureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    ns = argparse.Namespace(
        db=str(FIXTURE_DB),
        baseline="sess_cold_0001",
        candidate="sess_skill_0002",
        json=args.json,
        out=args.out,
    )
    # stderr, not stdout: `demo --json` must stay machine-readable.
    print(
        "# Recorded fixture. No Hermes install and no API key involved.\n"
        "# Same code path as `compare` against your own state.db.",
        file=sys.stderr,
    )
    return cmd_compare(ns)


def cmd_probe(args: argparse.Namespace) -> int:
    if args.write:
        written = injection.write_payloads(Path(args.write))
        print("Wrote probe files:")
        for p in written:
            print(f"  {p}")
        print(
            "\nNow point Hermes at that directory and ask it to summarise each "
            "file.\nThen run:  python main.py probe --session <session-id>"
        )
        return 0

    if args.session:
        with _open(args.db) as db:
            results = injection.scan_session(db, args.session)
        print(injection.render(results))
        return 1 if any(r.followed_the_file for r in results) else 0

    print("Give either --write <dir> or --session <id>", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hermes-ops-agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        help=f"Path to state.db (default: {default_db_path()})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sessions", help="list sessions")
    s.add_argument("--source", help="filter by source, e.g. cli or telegram")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_sessions)

    c = sub.add_parser("compare", help="compare two sessions")
    c.add_argument("baseline", help="session id of the cold run")
    c.add_argument("candidate", help="session id of the skill-present run")
    c.add_argument("--json", action="store_true", help="JSON instead of a table")
    c.add_argument("--out", help="also write the JSON report here")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo", help="compare the recorded fixture, no key needed")
    d.add_argument("--json", action="store_true")
    d.add_argument("--out")
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("probe", help="prompt-injection probe")
    pr.add_argument("--write", metavar="DIR", help="write payload files here")
    pr.add_argument("--session", help="scan this session for the canary")
    pr.set_defaults(func=cmd_probe)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StateDbError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

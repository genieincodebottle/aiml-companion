#!/usr/bin/env python3
"""Run this project without installing it, and without an API key.

`make` is not present on a default Windows install, and the Makefile targets
assume two API keys. This gives you one command that runs the whole graph:

    python run.py demo

Commands
--------
    demo        Research a topic through all eight nodes, and show the route.
    research    The same, on a topic you supply.
    trace       Just the node-by-node trace and the numbers, no report.
    test        Run the test suite.
    ui          Launch the Streamlit app.
    evaluate    Single-agent against multi-agent comparison.

Everything except a run with real keys uses a deterministic stand-in for the
model and for web search. The stand-in is a rule engine, not a model, and the
output says so.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _offline_notice() -> None:
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("No GOOGLE_API_KEY found, so this is an OFFLINE run.")
        print("Replies come from a deterministic rule engine, not a model.\n")


def _run_graph(query: str, verbose: bool):
    from src.agents.graph import build_graph

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(message)s"
    )
    app = build_graph()
    return app.invoke({"query": query})


def _summarise(result) -> None:
    trace = [t.get("agent") for t in (result.get("pipeline_trace") or [])]
    sources = result.get("sources") or []
    claims = result.get("key_claims") or []
    conflicts = result.get("conflicts") or []
    review = result.get("review") or {}

    _banner("HOW THE WORK ACTUALLY FLOWED")
    print("  " + " -> ".join(trace))
    print()
    print(f"  sub-topics planned      {len(result.get('sub_topics') or [])}")
    print(f"  researchers dispatched  {trace.count('researcher')} in parallel")
    print(f"  sources gathered        {len(sources)}")
    print(f"  claims extracted        {len(claims)}")
    print(f"  cross-source conflicts  {len(conflicts)}")
    print(f"  quality gate passed     {result.get('quality_passed')}")
    # `revision_count` is a draft VERSION counter, not a count of revisions:
    # the first draft sets it to 1. Reporting it as "revisions" would say 2
    # when the writer revised once.
    drafts = result.get("revision_count", 0)
    print(f"  drafts written          {drafts} ({max(drafts - 1, 0)} revision"
          f"{'' if drafts == 2 else 's'})")
    print(f"  reviewer score          {review.get('score', 'n/a')}")
    print(f"  tokens counted          {result.get('token_count', 0)}")

    if conflicts:
        _banner("THE DISAGREEMENT THE ANALYST FOUND")
        for c in conflicts:
            text = c.get("description") if isinstance(c, dict) else c
            print(f"  {text}")

    _banner("WHAT TO NOTICE")
    print("1. The researcher appears more than once in the trace. That is Send()")
    print("   fan-out: one researcher per sub-topic, running in parallel, with")
    print("   results merged by an operator.add reducer on `sources`.")
    print("2. The writer and reviewer BOTH appear twice. The reviewer failed the")
    print("   first draft, and a conditional edge sent the graph backwards. A")
    print("   pipeline that cannot go backwards is not a graph, it is a queue.")
    print("3. The conflict above was not resolved by picking a side. Both claims")
    print("   are true, and the disagreement is itself the finding.")
    print("4. Tokens are counted from real usage metadata, not estimated. When a")
    print("   provider reports nothing the count is 0 and the budget guardrail")
    print("   says it is blind, because a plausible guess would hide that.")


def cmd_demo(args) -> int:
    _banner("MULTI-AGENT RESEARCH: 7 agents, 8 graph nodes")
    _offline_notice()
    query = "What are the latest trends in AI agents?"
    print(f"Researching: {query}")
    result = _run_graph(query, args.verbose)

    report = result.get("final_report")
    if not report:
        print("\nNo report generated. Run with -v to see which node failed.",
              file=sys.stderr)
        return 1

    _summarise(result)
    _banner("THE REPORT")
    print(report)
    return 0


def cmd_research(args) -> int:
    query = " ".join(args.topic) if args.topic else "What are the latest trends in AI agents?"
    _offline_notice()
    print(f"Researching: {query}\n")
    result = _run_graph(query, args.verbose)
    report = result.get("final_report")
    if not report:
        print("No report generated.", file=sys.stderr)
        return 1
    print(report)
    return 0


def cmd_trace(args) -> int:
    _offline_notice()
    result = _run_graph("What are the latest trends in AI agents?", args.verbose)
    _summarise(result)
    return 0


def cmd_test(args) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=str(ROOT))


def cmd_ui(args) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=str(ROOT))


def cmd_evaluate(args) -> int:
    return subprocess.call([sys.executable, "-m", "evaluation.run_eval"], cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subs = parser.add_subparsers(dest="command")
    for name, fn, help_text in (
        ("demo", cmd_demo, "research a topic through all eight nodes, with the route"),
        ("research", cmd_research, "research a topic you supply"),
        ("trace", cmd_trace, "just the trace and the numbers"),
        ("test", cmd_test, "run the test suite"),
        ("ui", cmd_ui, "launch the Streamlit app"),
        ("evaluate", cmd_evaluate, "single-agent against multi-agent"),
    ):
        sub = subs.add_parser(name, help=help_text)
        sub.set_defaults(func=fn)
        sub.add_argument("-v", "--verbose", action="store_true", help="show agent logs")
        if name == "research":
            sub.add_argument("topic", nargs="*", help="what to research")

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

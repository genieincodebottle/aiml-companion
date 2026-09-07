#!/usr/bin/env python
"""One entry point for this project, on any operating system.

    python run.py doctor      check setup: env, Neo4j, model access
    python run.py test        unit tests (no API key, no database needed)
    python run.py ingest      build the knowledge graph  (--reset to wipe first)
    python run.py ask "..."   answer one question  (--strategy hybrid)
    python run.py compare "..."  run every strategy on one question, side by side
    python run.py eval        the golden-question benchmark  (--no-judge to skip
                              the LLM judge and run deterministic metrics only)
    python run.py app         launch the Streamlit UI
    python run.py cypher      print the Cypher cookbook with live row counts

`doctor` is the one to run first. Nearly every failure in a stack like this is
a setup failure, and the symptoms are misleading: a wrong Neo4j port looks like
an auth error, a placeholder API key looks like a model outage, and a
dimension mismatch looks like an empty corpus. `doctor` checks each of those
directly and tells you which one it is.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor() -> int:
    from src.config import ConfigError, get_config

    ok = True

    def check(label: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            print(f"  [ok]   {label}" + (f"  {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
            ok = False
            print(f"  [FAIL] {label}")
            for line in str(exc).splitlines():
                print(f"         {line}")

    print("\nChecking configuration\n")
    config = get_config()

    check("configs/base.yaml readable", lambda: f"model={config.llm['model']}")

    def _key() -> str:
        key = config.google_api_key
        return f"GOOGLE_API_KEY set ({key[:6]}...{key[-4:]})"
    check("Google API key", _key)

    def _neo4j() -> str:
        from src.graph.client import GraphClient
        with GraphClient(config) as client:
            client.verify()
            version = client.run(
                "CALL dbms.components() YIELD name, versions "
                "RETURN name + ' ' + versions[0] AS v"
            )
            return version[0]["v"] if version else "connected"
    check(f"Neo4j at {config.neo4j.uri}", _neo4j)

    def _model() -> str:
        from src.llm import LLMClient
        client = LLMClient(config)
        reply = client.generate("Reply with the single word: ready")
        return f"generation ok ({reply[:20]!r})"
    check(f"LLM model '{config.llm['model']}'", _model)

    def _embed() -> str:
        from src.llm import LLMClient
        client = LLMClient(config)
        vector = client.embed_query("supply chain risk")
        expected = config.embedding["dimensions"]
        if len(vector) != expected:
            raise RuntimeError(
                f"embedding model returned {len(vector)} dimensions but "
                f"configs/base.yaml declares {expected}. The vector index is "
                "built from that number, and a mismatch makes every search "
                "return nothing without erroring."
            )
        return f"{len(vector)} dimensions"
    check(f"Embedding model '{config.embedding['model']}'", _embed)

    def _graph() -> str:
        from src.graph.client import GraphClient
        with GraphClient(config) as client:
            counts = client.counts()
            if not counts:
                raise RuntimeError(
                    "The database is empty. Run `python run.py ingest` to build "
                    "the graph (takes a few minutes and about 40 LLM calls)."
                )
            entities = sum(v for k, v in counts.items() if k.startswith("node:"))
            rels = sum(v for k, v in counts.items() if k.startswith("rel:"))
            return f"{entities} nodes, {rels} relationships"
    check("Knowledge graph populated", _graph)

    print("\n" + ("All checks passed." if ok else "Some checks failed - see above."))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def cmd_ingest(reset: bool) -> int:
    from src.ingest.pipeline import ingest

    def progress(message: str, fraction: float) -> None:
        print(f"  [{fraction * 100:5.1f}%] {message}")

    print("\nBuilding the knowledge graph\n")
    report = ingest(reset=reset, progress=progress)
    print("\n" + json.dumps(report.as_dict(), indent=2))

    if report.dropped:
        print(f"\n{len(report.dropped)} extraction outputs were dropped by "
              f"validation. First 10:")
        for item in report.dropped[:10]:
            print(f"  - {item}")
        print("Dropped items are not errors. They are the extractor proposing "
              "something outside the closed vocabulary, and the validator "
              "refusing it. A count of zero here would be more suspicious.")
    return 0


# ---------------------------------------------------------------------------
# ask / compare
# ---------------------------------------------------------------------------
def _run_question(question: str, strategies: list[str], show_trace: bool) -> int:
    """Answer one question through the SAME service layer the API uses.

    This matters more than it looks. If the CLI called the retriever directly it
    would skip the guardrails, the budget cap and the audit log, so `run.py ask`
    and `POST /api/ask` would be two subtly different systems and a bug fixed in
    one would survive in the other. Both go through QAService: one pipeline,
    two front doors.
    """
    from api.deps import build_services      # composition root, not a web import
    from src.guardrails import GuardrailViolation
    from src.retrieval.strategies import STRATEGY_LABELS

    services = build_services()
    try:
        for strategy in strategies:
            print("\n" + "=" * 78)
            print(f"{STRATEGY_LABELS[strategy]}")
            print("=" * 78)
            try:
                bundle = services.qa.ask(question, strategy, caller="cli",
                                         include_trace=show_trace)
            except GuardrailViolation as exc:
                print(f"\nBlocked by a guardrail ({exc.kind}): {exc}\n")
                continue

            if show_trace and bundle.retrieval.trace:
                print("\nHow it retrieved:")
                for i, line in enumerate(bundle.retrieval.trace, 1):
                    print(f"  {i}. {line}")

            summary = bundle.answer.as_dict()
            print(f"\nRetrieved: {summary['text_chunks']} chunks from "
                  f"{summary['documents']} documents, "
                  f"{summary['graph_facts']} derived graph facts, "
                  f"{summary['entities']} entities, "
                  f"max {summary['max_hops']} hops")
            print(f"Latency: {summary['retrieval_ms']}ms retrieval + "
                  f"{summary['generation_ms']}ms generation  "
                  f"| cost ${bundle.usage['estimated_usd']:.5f}")

            validation = bundle.answer.validation
            if validation and not validation.get("ok", True):
                print("\nOutput guardrails flagged this answer:")
                for warning in validation.get("warnings", []):
                    print(f"  [{warning['severity']}] {warning['kind']}: "
                          f"{warning['detail']}")
            print(f"\n{bundle.answer.text}\n")
    finally:
        services.graph_client.close()
    return 0


# ---------------------------------------------------------------------------
# cypher cookbook
# ---------------------------------------------------------------------------
def cmd_cypher() -> int:
    """Print the graph's shape and a few worked queries against live data.

    This exists because reading Cypher in a file teaches less than seeing it
    return rows from a database you just built."""
    from src.config import get_config
    from src.graph import queries
    from src.graph.client import GraphClient
    from src.graph.schema import index_status

    config = get_config()
    with GraphClient(config) as client:
        client.verify()

        print("\nGRAPH CENSUS\n" + "-" * 40)
        for label, count in sorted(client.counts().items()):
            print(f"  {label:<28} {count:>5}")

        print("\nINDEXES AND CONSTRAINTS\n" + "-" * 40)
        for row in index_status(client):
            print(f"  {row['name']:<28} {row['type']:<10} {row['state']}")

        print("\nWORKED QUERY 1: products exposed to a disruption in Kaohsiung")
        print("-" * 70)
        rows = client.run(queries.PRODUCTS_EXPOSED_TO_LOCATION,
                          location_key="location:kaohsiung", limit=25)
        if not rows:
            print("  (no rows - has the graph been ingested?)")
        for row in rows:
            chain = " -> ".join(reversed(row["dependency_chain"]))
            print(f"  {row['product']:<26} via {row['component']:<34} "
                  f"tier {row['tier_depth']}  [{chain}]")

        print("\nWORKED QUERY 2: sole-sourced parts whose supplier has a finding")
        print("-" * 70)
        for row in client.run(queries.SOLE_SOURCE_WITH_FINDINGS):
            print(f"  {row['supplier']:<26} {row['component']:<34} "
                  f"{', '.join(row['products'])}")

        print("\nWORKED QUERY 3: supplier criticality by product fan-out")
        print("-" * 70)
        for row in client.run(queries.SUPPLIER_CRITICALITY, limit=10):
            print(f"  {row['supplier']:<32} {row['products_at_risk']} products  "
                  f"{', '.join(row['products'])}")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show progress logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check environment, database and model access")
    sub.add_parser("test", help="run unit tests (no API key or database needed)")
    sub.add_parser("security", help="demonstrate the guardrails (no API key needed)")
    sub.add_parser("api", help="launch the FastAPI backend")
    sub.add_parser("cypher", help="print the graph census and worked queries")
    sub.add_parser("app", help="launch the Streamlit UI")

    p_ingest = sub.add_parser("ingest", help="build the knowledge graph")
    p_ingest.add_argument("--reset", action="store_true",
                          help="wipe the database first (use after changing the schema)")

    p_ask = sub.add_parser("ask", help="answer one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--strategy", default="hybrid",
                       help="vector | keyword | classic | graph | hybrid")
    p_ask.add_argument("--no-trace", action="store_true")

    p_compare = sub.add_parser("compare", help="run every strategy on one question")
    p_compare.add_argument("question")

    p_eval = sub.add_parser("eval", help="run the golden-question benchmark")
    p_eval.add_argument("--no-judge", action="store_true",
                        help="skip the LLM judge; deterministic metrics only")
    p_eval.add_argument("--strategy", action="append", default=None,
                        help="limit to one strategy (repeatable)")
    p_eval.add_argument("--question", action="append", default=None,
                        help="limit to one question id (repeatable)")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])

    if args.command == "security":
        from src.security_demo import run as run_security
        return run_security()

    if args.command == "api":
        return subprocess.call([
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", "8000",
        ])

    if args.command == "app":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
        )

    # Everything below needs configuration, so import lazily and turn the
    # setup errors into instructions rather than tracebacks.
    from src.config import ConfigError

    try:
        if args.command == "doctor":
            return cmd_doctor()
        if args.command == "ingest":
            return cmd_ingest(args.reset)
        if args.command == "cypher":
            return cmd_cypher()
        if args.command == "ask":
            return _run_question(args.question, [args.strategy], not args.no_trace)
        if args.command == "compare":
            from src.retrieval.strategies import STRATEGIES
            return _run_question(args.question, list(STRATEGIES), True)
        if args.command == "eval":
            from src.config import get_config
            from src.evaluate import evaluate, format_report

            def progress(message: str, fraction: float) -> None:
                print(f"  [{fraction * 100:5.1f}%] {message}")

            print("\nRunning the golden-question benchmark. This makes a few "
                  "hundred model calls and takes several minutes.\n")
            report = evaluate(
                strategies=args.strategy, judge=not args.no_judge,
                question_ids=args.question, progress=progress,
            )
            print("\n" + format_report(report))
            out = get_config().root / "artifacts" / "evaluation.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            print(f"\nFull results written to {out}")
            return 0
    except ConfigError as exc:
        print(f"\nSetup incomplete:\n\n{exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        from src.graph.client import GraphUnavailable
        if isinstance(exc, GraphUnavailable):
            print(f"\n{exc}\n", file=sys.stderr)
            return 1
        raise

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

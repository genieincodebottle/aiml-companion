"""Entry point for `python -m src.agents`, `python run.py research` and `make run`."""

import logging
import sys

from src.agents.graph import build_graph


def main() -> int:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest trends in AI agents?"

    # The agents log their progress at INFO, and nothing configured logging, so
    # a reader watching eight nodes run saw only the final report. Python's
    # last-resort handler covers WARNING and above, which is why errors appeared
    # and progress did not.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Researching: {query}\n")

    app = build_graph()
    result = app.invoke({"query": query})

    report = result.get("final_report")
    if not report:
        # This used to print "No report generated." and return 0. Every agent
        # catches its own exceptions, so a run in which all eight nodes failed
        # still reached this line and still reported success to the shell.
        errors = result.get("errors") or []
        print("\nNo report generated.", file=sys.stderr)
        for err in errors[:5]:
            print(f"  {err}", file=sys.stderr)
        print(
            "\nEvery agent catches its own errors, so the pipeline completes even "
            "when nothing worked. Run with RESEARCH_OFFLINE=1 to check the graph "
            "itself without keys.",
            file=sys.stderr,
        )
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

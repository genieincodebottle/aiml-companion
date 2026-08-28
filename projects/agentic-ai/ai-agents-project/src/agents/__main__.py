"""Entry point for `python -m src.agents` and `make run`."""

import sys
from src.agents.graph import build_graph


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest trends in AI agents?"
    print(f"Researching: {query}\n")

    app = build_graph()
    result = app.invoke({"query": query})

    report = result.get("final_report", "No report generated.")
    print(report)


if __name__ == "__main__":
    main()

"""Experiment 3. A full agentic loop, with ATEM tool calls.

    uv run python scripts/03_tool_loop.py

Three questions, each routed to a different tool. Watch the recipient on each
message: `to=self` is private, `to=search_docs` is a tool request, `to=user` is
the answer. The loop stops on the first `to=user` message and on nothing else.

The last question is the interesting one. It asks for a number the model cannot
know without calling a tool, and the tool computes it from the architecture
constants, so the answer is derived rather than recalled.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets `python scripts/xx.py` work as well as `uv run python scripts/xx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glimmer.agent import run_agent
from glimmer.client import build_client

QUESTIONS = [
    "Which tokens should I use as stop tokens, and which one is a trap?",
    "What is 131072 / 4096?",
    "How much KV cache does a 131072 token context need?",
]


def main() -> None:
    client = build_client()
    for question in QUESTIONS:
        print("\n" + "=" * 78)
        print(f"Q: {question}")
        run = run_agent(client, question, reasoning_strength="medium", verbose=True)
        print(f"\n  answer: {run.answer}")
        print(f"  {run.tool_calls} tool call(s), stopped because it {run.stopped_because}")
    client.close()


if __name__ == "__main__":
    main()

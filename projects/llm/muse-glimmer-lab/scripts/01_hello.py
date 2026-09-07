"""Experiment 1. Hello world, and the three channels.

    uv run python scripts/01_hello.py

Runs offline by default. Point it at a real server with

    GLIMMER_MODE=live GLIMMER_BASE_URL=http://127.0.0.1:8080/v1 \
        uv run python scripts/01_hello.py

What to look for: the raw string is not prose. It is three messages, each with
its own recipient, and the parser is what turns that into something you can
use. Read the raw block before reading the parsed one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets `python scripts/xx.py` work as well as `uv run python scripts/xx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glimmer import channels
from glimmer.client import Message, build_client, system_prompt
from glimmer.tools import describe_tools

QUESTION = "What sliding window does Muse Glimmer use on its local layers?"


def main() -> None:
    client = build_client()
    response = client.complete(
        [
            Message("system", system_prompt("medium", describe_tools())),
            Message("user", QUESTION),
        ]
    )

    print(f"mode: {response.mode}\n")
    print("-- raw generation " + "-" * 55)
    print(response.raw)

    turn = channels.parse(response.raw)
    print("\n-- parsed into channels " + "-" * 49)
    for message in turn.messages:
        kind = "reasoning" if message.is_reasoning else "final" if message.is_final else "tool call"
        print(f"\n  [{kind}]  to={message.recipient}  ends with {message.terminator}")
        print("  " + "\n  ".join(message.content.splitlines()))

    print("\n-- what the loop would do next " + "-" * 42)
    print(f"  wants a tool: {turn.wants_tool}")
    print(f"  visible answer so far: {turn.final or '(none yet)'}")
    client.close()


if __name__ == "__main__":
    main()

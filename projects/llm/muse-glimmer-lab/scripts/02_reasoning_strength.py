"""Experiment 2. Change one input and watch the output change.

    uv run python scripts/02_reasoning_strength.py

Reasoning strength is Glimmer's one big latency knob. It does not change the
weights or the sampler. It changes how much the model writes into the `to=self`
channel before it addresses the user, and that text costs real tokens and real
seconds.

The table prints reasoning tokens against answer tokens for each setting. The
ratio is the point. At xhigh most of what you pay for never reaches the user.

The trap worth knowing, documented in client.system_prompt: on a served
endpoint the chat template appends its own reasoning directive after yours and
defaults to high. If live mode shows four settings that all behave the same,
that is what happened, and the fix is the template parameter rather than the
system prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets `python scripts/xx.py` work as well as `uv run python scripts/xx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glimmer import channels
from glimmer.client import Message, build_client, system_prompt
from glimmer.config import REASONING_STRENGTHS
from glimmer.tools import describe_tools

QUESTION = "Why does Muse Glimmer put a global attention layer every fourth layer?"


def main() -> None:
    client = build_client()
    print(f"{'strength':<10}{'reasoning tok':>15}{'answer tok':>13}{'ratio':>9}")
    print("-" * 47)

    for strength in REASONING_STRENGTHS:
        turn = channels.parse(
            client.complete(
                [
                    Message("system", system_prompt(strength, describe_tools())),
                    Message("user", QUESTION),
                ]
            ).raw
        )
        # Rough token counts. Four characters per token is close enough to
        # compare settings against each other, which is all this table does.
        reasoning = len(turn.reasoning) // 4
        answer = len(turn.final or turn.tool_calls[0].content if turn.tool_calls else "") // 4
        ratio = f"{reasoning / answer:.1f}x" if answer else "-"
        print(f"{strength:<10}{reasoning:>15}{answer:>13}{ratio:>9}")

    print(
        "\nHigher strength buys accuracy on hard agentic and coding work, and Meta\n"
        "recommends high or xhigh there. For a chat reply it is mostly latency."
    )
    client.close()


if __name__ == "__main__":
    main()

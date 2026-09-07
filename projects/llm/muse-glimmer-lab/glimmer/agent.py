"""The agent loop, in about forty lines and no framework.

The loop is the whole of what Muse Glimmer was built for, so it is worth seeing
without a library in the way.

    ask the model
    parse the turn into channels
    if it addressed a tool, run the tool and feed the result back
    if it addressed the user, stop

Everything Glimmer-specific lives in the parsing, not in the loop. Swap in a
JSON tool-calling model and the loop is unchanged; only channels.parse and
atem.parse_tool_calls come out.

Two guards that a toy loop usually skips and a real one cannot.

`max_steps` bounds the run. A model that keeps calling a failing tool will keep
calling it forever, and forever on a local GPU is cheap in money and expensive
in an afternoon.

Tool errors come back as text, not exceptions. See tools.run_tool for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import atem, channels
from .client import Message, system_prompt
from .tools import SCHEMAS, describe_tools, run_tool


@dataclass
class Step:
    index: int
    reasoning: str
    tool_name: str = ""
    tool_arguments: dict = field(default_factory=dict)
    tool_result: str = ""
    final: str = ""


@dataclass
class Run:
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    stopped_because: str = ""

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name)


def run_agent(
    client,
    question: str,
    *,
    reasoning_strength: str = "high",
    max_steps: int = 6,
    verbose: bool = False,
) -> Run:
    """Drive one question to an answer, printing the channels as they arrive."""
    messages = [
        Message("system", system_prompt(reasoning_strength, describe_tools())),
        Message("user", question),
    ]
    run = Run()

    for index in range(1, max_steps + 1):
        turn = channels.parse(client.complete(messages).raw)
        step = Step(index=index, reasoning=turn.reasoning)

        if verbose:
            print(f"\n  step {index}")
            if turn.reasoning:
                print(f"    to=self   {_clip(turn.reasoning)}")

        if turn.wants_tool:
            call_message = turn.tool_calls[0]
            calls = atem.parse_tool_calls(call_message.content)
            if not calls:
                # The model addressed a tool but emitted nothing parseable.
                # Tell it so, in the transcript, and let it retry.
                messages.append(Message("tool", "Error: no readable tool call found."))
                step.tool_result = "unparseable"
                run.steps.append(step)
                continue

            call = calls[0]
            arguments = atem.coerce_arguments(call, SCHEMAS.get(call.name, {}).get("properties", {}))
            result = run_tool(call.name, arguments)

            step.tool_name = call.name
            step.tool_arguments = arguments
            step.tool_result = result
            if verbose:
                print(f"    to={call.name}  {arguments}")
                print(f"    result    {_clip(result)}")

            # The tool call and its result both go back, so the model sees what
            # it asked for alongside what it got.
            messages.append(Message("assistant", call_message.content))
            messages.append(Message("tool", result))
            run.steps.append(step)
            continue

        step.final = turn.final
        run.steps.append(step)
        run.answer = turn.final
        run.stopped_because = "answered"
        if verbose:
            print(f"    to=user   {_clip(turn.final)}")
        return run

    run.stopped_because = f"hit max_steps={max_steps}"
    return run


def _clip(text: str, width: int = 88) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "..."

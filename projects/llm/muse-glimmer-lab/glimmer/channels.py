"""Parsing Muse Glimmer's channel-scoped output.

Most chat models wrap their thinking in `<think>...</think>` and everything
else is the answer. Muse Glimmer does something different, and if you treat it
like a `<think>` model you get an empty string back and no idea why.

Glimmer tags every message with a recipient. One turn of generation can emit
several messages, each addressed differently:

    <|start|>assistant to=self<|message|>    ...private reasoning...   <|eom|>
    <|start|>assistant to=search<|message|>  ...an ATEM tool call...   <|eom|>
    <|start|>assistant to=user<|message|>    ...the visible reply...   <|eot|>

Two things follow from that shape, and both bite people.

`<|eom|>` means end of message, not end of turn. A tool call ends with it and
the turn continues after the tool result comes back. `<|eot|>` and
`<|end_of_text|>` are the only real stop tokens. Stop on `<|eom|>` and the
model appears to answer with a tool call and then fall silent.

The markers are special tokens, so a decoder running with the usual
`skip_special_tokens=True` deletes the very delimiters that separate reasoning
from the answer. The three channels collapse into one run-on string. This is
why vLLM's `--reasoning-parser muse_glimmer` forces `skip_special_tokens=False`.

This module is the parser, written out longhand so you can see the state
machine rather than trusting a flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

START = "<|start|>"
MESSAGE = "<|message|>"
END_OF_MESSAGE = "<|eom|>"
END_OF_TURN = "<|eot|>"
END_OF_TEXT = "<|end_of_text|>"

#: The only tokens that should ever terminate generation. Pass these to
#: llama.cpp's `--stop` or vLLM's `stop` and leave `<|eom|>` out.
STOP_TOKENS = (END_OF_TURN, END_OF_TEXT)

# <|start|>assistant to=user<|message|>  ->  role "assistant", recipient "user".
# The recipient is optional; a bare <|start|>assistant<|message|> means the user.
_HEADER = re.compile(
    r"<\|start\|>\s*(?P<role>[A-Za-z0-9_.-]+)"
    r"(?:\s+to=(?P<recipient>[A-Za-z0-9_.\-]+))?\s*<\|message\|>"
)


@dataclass
class Message:
    """One channel-scoped message inside a single assistant turn."""

    role: str
    recipient: str
    content: str
    terminator: str = ""

    @property
    def is_reasoning(self) -> bool:
        """Addressed to itself, so it is private chain of thought.

        Do not show this to a user and do not feed it back as if the model had
        said it out loud. It is the model's scratch pad.
        """
        return self.recipient == "self"

    @property
    def is_final(self) -> bool:
        """Addressed to the user, so this is the visible answer."""
        return self.recipient == "user"

    @property
    def is_tool_call(self) -> bool:
        """Addressed to anything else, so the recipient names a tool."""
        return not self.is_reasoning and not self.is_final


@dataclass
class Turn:
    """Everything the model emitted between one prompt and one stop token."""

    messages: list[Message] = field(default_factory=list)

    @property
    def reasoning(self) -> str:
        return "\n".join(m.content for m in self.messages if m.is_reasoning)

    @property
    def final(self) -> str:
        return "\n".join(m.content for m in self.messages if m.is_final)

    @property
    def tool_calls(self) -> list[Message]:
        return [m for m in self.messages if m.is_tool_call]

    @property
    def wants_tool(self) -> bool:
        """True when the turn ended on a tool call rather than an answer.

        The agent loop in agent.py keys off this. A turn can contain reasoning
        and a tool call and no final message at all, which is the normal shape
        of a working step in an agentic run.
        """
        return bool(self.tool_calls) and not self.final


def parse(raw: str) -> Turn:
    """Split one raw generation into its channel-scoped messages.

    The parser is forgiving in one specific way. A response that begins mid
    message, with no leading `<|start|>`, is treated as an implicit
    `assistant to=user` message. Servers that strip the opening header are
    common enough that failing on it would be unhelpful, and a plain
    non-Glimmer completion then parses as a single final message.

    >>> t = parse("<|start|>assistant to=self<|message|>think<|eom|>"
    ...           "<|start|>assistant to=user<|message|>hi<|eot|>")
    >>> t.reasoning, t.final
    ('think', 'hi')
    """
    turn = Turn()
    if not raw:
        return turn

    matches = list(_HEADER.finditer(raw))

    # No headers at all. Treat the whole thing as the visible answer.
    if not matches:
        content, terminator = _split_terminator(raw)
        if content.strip():
            turn.messages.append(Message("assistant", "user", content.strip(), terminator))
        return turn

    # Text before the first header is an unheaded leading message.
    preamble = raw[: matches[0].start()]
    if preamble.strip():
        content, terminator = _split_terminator(preamble)
        turn.messages.append(Message("assistant", "user", content.strip(), terminator))

    for i, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body, terminator = _split_terminator(raw[body_start:body_end])
        turn.messages.append(
            Message(
                role=match.group("role"),
                # A missing `to=` means the user. That is the same default the
                # chat template applies when it renders a plain reply.
                recipient=match.group("recipient") or "user",
                content=body.strip(),
                terminator=terminator,
            )
        )
    return turn


def _split_terminator(chunk: str) -> tuple[str, str]:
    """Strip a trailing end marker and report which one it was.

    Which marker ended a message is not decoration. `<|eom|>` on the last
    message means the model is waiting for a tool result, and `<|eot|>` means
    it considers the turn finished.
    """
    for token in (END_OF_MESSAGE, END_OF_TURN, END_OF_TEXT):
        index = chunk.find(token)
        if index != -1:
            return chunk[:index], token
    return chunk, ""


def render(messages: list[Message]) -> str:
    """Rebuild the wire format from parsed messages.

    Round-tripping is the cheapest test that the parser understood the format,
    and tests/test_channels.py uses it exactly that way.
    """
    out = []
    for message in messages:
        terminator = message.terminator or (
            END_OF_TURN if message.is_final else END_OF_MESSAGE
        )
        out.append(
            f"{START}{message.role} to={message.recipient}{MESSAGE}"
            f"{message.content}{terminator}"
        )
    return "".join(out)

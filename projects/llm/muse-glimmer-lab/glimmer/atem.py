"""ATEM, the XML-shaped tool-call format Muse Glimmer emits.

Nearly every other tool-calling model emits JSON. Glimmer emits this:

    <atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Bengaluru</atem:parameter>
    <atem:parameter name="units">metric</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

The choice looks like a step backwards until you think about what a language
model has to do to produce valid JSON. A JSON call is only well formed once the
final brace lands, and every string value inside it has to be escaped, so a
snippet of code passed as an argument turns into a thicket of `\\n` and `\\"`.
One wrong escape invalidates the whole object. Models drop tokens under long
agentic runs, and the failure is all-or-nothing.

ATEM degrades more gently. Each parameter is delimited by its own closing tag,
so a malformed value damages one argument rather than the call. Argument values
are raw text between tags, which means a shell command or a Python function
goes in verbatim with nothing to escape. That matters for a model whose whole
design goal is hours-long tool use, and it is why the format is worth
understanding rather than just switching a parser on.

The trade-off is real and worth stating. ATEM has no types. Everything comes
out a string, so the caller has to coerce against the tool schema, and
`coerce_arguments` below is where that happens. JSON carries its own types.

This parser is deliberately regex-based rather than a real XML parse, because
the model's output is frequently not well-formed XML. An unescaped `<` inside a
parameter value is normal and an XML parser would reject the document.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_INVOKE = re.compile(
    r"<atem:invoke\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<body>.*?)</atem:invoke>",
    re.DOTALL,
)
_PARAM = re.compile(
    r"<atem:parameter\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<value>.*?)</atem:parameter>",
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Pull every ATEM invocation out of a message body.

    One `<atem:function_calls>` block can hold several `<atem:invoke>`
    elements, which is how Glimmer requests two independent lookups in one
    step. The wrapper is not required by this parser, so a bare `invoke`
    still parses.

    >>> calls = parse_tool_calls(
    ...     '<atem:invoke name="add">'
    ...     '<atem:parameter name="a">2</atem:parameter>'
    ...     '<atem:parameter name="b">3</atem:parameter>'
    ...     '</atem:invoke>')
    >>> calls[0].name, calls[0].arguments
    ('add', {'a': '2', 'b': '3'})
    """
    calls: list[ToolCall] = []
    for invoke in _INVOKE.finditer(text):
        arguments = {
            param.group("name"): param.group("value").strip()
            for param in _PARAM.finditer(invoke.group("body"))
        }
        calls.append(ToolCall(name=invoke.group("name"), arguments=arguments))
    return calls


def coerce_arguments(call: ToolCall, schema: dict[str, Any]) -> dict[str, Any]:
    """Turn ATEM's all-strings into the types a tool actually wants.

    This is the tax ATEM charges for being robust. A JSON tool call arrives
    with `2` already an integer; here it arrives as the string `"2"` and the
    schema is the only thing that knows better.

    `schema` is the JSON-Schema `properties` object for the tool. Anything the
    schema does not mention is passed through untouched, so an unexpected
    argument surfaces as a tool error rather than vanishing.
    """
    out: dict[str, Any] = {}
    for key, raw in call.arguments.items():
        spec = schema.get(key)
        if not spec:
            out[key] = raw
            continue
        kind = spec.get("type", "string")
        try:
            if kind == "integer":
                out[key] = int(raw)
            elif kind == "number":
                out[key] = float(raw)
            elif kind == "boolean":
                out[key] = raw.strip().lower() in ("true", "1", "yes")
            elif kind in ("object", "array"):
                # Nested structure is the one place ATEM falls back to JSON.
                out[key] = json.loads(raw)
            else:
                out[key] = raw
        except (ValueError, json.JSONDecodeError):
            # Keep the raw string. The tool will reject it with a message the
            # model can read and retry against, which is a better outcome than
            # a traceback that ends the run.
            out[key] = raw
    return out


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Build an ATEM block. Used by the offline client and by the tests."""
    params = "\n".join(
        f'<atem:parameter name="{key}">{value}</atem:parameter>'
        for key, value in arguments.items()
    )
    return (
        "<atem:function_calls>\n"
        f'<atem:invoke name="{name}">\n'
        f"{params}\n"
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )

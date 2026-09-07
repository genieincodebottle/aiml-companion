"""A three-tool catalogue, with no network calls and no keys.

The tools are deliberately dull. What you are here to watch is the loop, the
channel framing, and the ATEM parsing, and a tool that can fail for its own
reasons makes those harder to see.

`SCHEMAS` is standard JSON Schema. Glimmer is prompted with it, and
atem.coerce_arguments uses it to turn ATEM's strings back into numbers, which
is the one place the XML format needs help that JSON would not have needed.
"""

from __future__ import annotations

from typing import Any, Callable

SCHEMAS: dict[str, dict[str, Any]] = {
    "search_docs": {
        "description": "Search the local documentation index for a phrase.",
        "properties": {
            "query": {"type": "string", "description": "Words to look for."},
            "top_k": {"type": "integer", "description": "How many hits to return."},
        },
        "required": ["query"],
    },
    "calculate": {
        "description": "Evaluate a small arithmetic expression.",
        "properties": {
            "expression": {"type": "string", "description": "For example 17 * 3 + 2."},
        },
        "required": ["expression"],
    },
    "kv_cache_gib": {
        "description": "Compute Muse Glimmer's KV cache size for a context length.",
        "properties": {
            "context": {"type": "integer", "description": "Context length in tokens."},
            "use_sliding_window": {
                "type": "boolean",
                "description": "False bills every layer as full attention.",
            },
        },
        "required": ["context"],
    },
}

# A stand-in corpus so search_docs returns something real to reason over.
_DOCS = [
    ("gqa", "Muse Glimmer uses 32 query heads and 2 key/value heads, a 16:1 GQA ratio."),
    ("window", "Local layers use a 2,048-token sliding window. Every fourth layer is global."),
    ("rope", "RoPE theta is 500,000 and is applied to local layers only."),
    ("stop", "Stop on <|eot|> and <|end_of_text|>. Never stop on <|eom|>."),
    ("dflash", "The DFlash drafter proposes 16 tokens per forward pass and the target verifies them."),
    ("quant", "The kquant-17gb build is 16.8 GB on disk and targets a 24 GB card."),
]


def search_docs(query: str, top_k: int = 3) -> str:
    """Substring search, scored by how many query words a line contains."""
    words = [w for w in query.lower().split() if len(w) > 2]
    scored = []
    for key, line in _DOCS:
        hits = sum(1 for w in words if w in line.lower() or w in key)
        if hits:
            scored.append((hits, line))
    scored.sort(reverse=True, key=lambda pair: pair[0])
    if not scored:
        return "No matches."
    return "\n".join(line for _, line in scored[:top_k])


def calculate(expression: str) -> str:
    """Arithmetic only.

    `eval` on model output would be a remote code execution hole, so the
    expression is checked against a whitelist of characters first and
    evaluated with no builtins. This is the smallest honest version of the
    rule that a tool must never trust its arguments, and an agentic model that
    runs for hours will eventually hand you something strange.
    """
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "Error: only digits and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except (SyntaxError, ZeroDivisionError, ValueError) as exc:
        return f"Error: {exc}"


def kv_cache_gib(context: int, use_sliding_window: bool = True) -> str:
    from .memory import kv_cache_bytes

    breakdown = kv_cache_bytes(context, use_sliding_window=use_sliding_window)
    return (
        f"{breakdown.total_gib:.2f} GiB "
        f"({breakdown.global_layers} global layers, {breakdown.local_layers} local)"
    )


REGISTRY: dict[str, Callable[..., str]] = {
    "search_docs": search_docs,
    "calculate": calculate,
    "kv_cache_gib": kv_cache_gib,
}


def describe_tools() -> str:
    """Render the catalogue for the system prompt.

    Glimmer's chat template has its own tool section, and a served endpoint
    with `--enable-auto-tool-choice` will render tools for you from the
    `tools` field of the request. This function is for the offline path and
    for seeing plainly what the model is being told.
    """
    lines = []
    for name, schema in SCHEMAS.items():
        args = ", ".join(
            f"{key}: {spec['type']}" for key, spec in schema["properties"].items()
        )
        lines.append(f"- {name}({args}) - {schema['description']}")
    return "\n".join(lines)


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch, with failures returned as text rather than raised.

    A raised exception ends the agent run. A returned error string goes back
    into the transcript as a tool result, the model reads it, and it gets to
    try something else. For a model built for hours-long sessions that
    difference is most of what reliability means.
    """
    func = REGISTRY.get(name)
    if func is None:
        return f"Error: no tool named {name!r}. Available: {', '.join(REGISTRY)}."
    try:
        return func(**arguments)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"

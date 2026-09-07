from glimmer import atem
from glimmer.atem import ToolCall
from glimmer.tools import SCHEMAS


def test_parses_name_and_arguments():
    calls = atem.parse_tool_calls(
        "<atem:function_calls>"
        '<atem:invoke name="search_docs">'
        '<atem:parameter name="query">sliding window</atem:parameter>'
        '<atem:parameter name="top_k">3</atem:parameter>'
        "</atem:invoke>"
        "</atem:function_calls>"
    )
    assert len(calls) == 1
    assert calls[0].name == "search_docs"
    assert calls[0].arguments == {"query": "sliding window", "top_k": "3"}


def test_several_invocations_in_one_block():
    calls = atem.parse_tool_calls(
        "<atem:function_calls>"
        '<atem:invoke name="a"><atem:parameter name="x">1</atem:parameter></atem:invoke>'
        '<atem:invoke name="b"><atem:parameter name="y">2</atem:parameter></atem:invoke>'
        "</atem:function_calls>"
    )
    assert [c.name for c in calls] == ["a", "b"]


def test_values_need_no_escaping():
    """The property that makes ATEM worth the loss of types.

    A JSON tool call would need every quote and newline in this escaped, and
    one bad escape invalidates the entire object.
    """
    code = 'def f():\n    return {"a": 1} < 2 & 3'
    calls = atem.parse_tool_calls(
        f'<atem:invoke name="run"><atem:parameter name="src">{code}</atem:parameter></atem:invoke>'
    )
    assert calls[0].arguments["src"] == code


def test_coercion_uses_the_schema():
    call = ToolCall("kv_cache_gib", {"context": "131072", "use_sliding_window": "false"})
    coerced = atem.coerce_arguments(call, SCHEMAS["kv_cache_gib"]["properties"])
    assert coerced == {"context": 131072, "use_sliding_window": False}


def test_bad_value_survives_as_a_string():
    """A tool then returns a readable error the model can retry against,
    which beats a traceback that ends the run."""
    call = ToolCall("kv_cache_gib", {"context": "lots"})
    coerced = atem.coerce_arguments(call, SCHEMAS["kv_cache_gib"]["properties"])
    assert coerced["context"] == "lots"


def test_unknown_argument_passes_through():
    call = ToolCall("calculate", {"expression": "1+1", "surprise": "x"})
    coerced = atem.coerce_arguments(call, SCHEMAS["calculate"]["properties"])
    assert coerced["surprise"] == "x"


def test_render_round_trips():
    rendered = atem.render_tool_call("calculate", {"expression": "17 * 3"})
    assert atem.parse_tool_calls(rendered)[0].arguments == {"expression": "17 * 3"}


def test_no_calls_in_plain_text():
    assert atem.parse_tool_calls("no tools here") == []

"""
Tests for the Tools package (tool selector, Wikipedia API structure).
Run: pytest tests/test_tools.py -v
"""
import pytest


def test_tool_selector_web_query():
    from src.tools.tool_selector import select_tools
    tools = select_tools("latest AI agent frameworks 2025")
    assert "tavily" in tools
    assert "wikipedia" not in tools  # "latest" signals web-only


def test_tool_selector_factual_query():
    from src.tools.tool_selector import select_tools
    tools = select_tools("what is retrieval augmented generation")
    assert "tavily" in tools
    assert "wikipedia" in tools  # "what is" signals Wikipedia


def test_tool_selector_mixed_query():
    from src.tools.tool_selector import select_tools
    tools = select_tools("what is the latest in transformer architecture 2025")
    assert "tavily" in tools
    assert "wikipedia" in tools  # Both signals present


def test_tool_selector_always_includes_tavily():
    from src.tools.tool_selector import select_tools
    tools = select_tools("random query with no keywords")
    assert "tavily" in tools


def test_tool_selector_definition_query():
    from src.tools.tool_selector import select_tools
    tools = select_tools("define gradient descent")
    assert "wikipedia" in tools

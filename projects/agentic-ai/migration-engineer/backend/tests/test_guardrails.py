"""Guardrail hooks: path containment, test protection, secret redaction."""

from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.guardrails.hooks import HookManager
from src.worktree import make_worktree


def _hooks() -> HookManager:
    wt: Path = make_worktree("billing-service", "guard-job", get_settings().data_dir)
    return HookManager(worktree=wt)


def test_allows_ordinary_source_write():
    d = _hooks().pre_tool_use("write_file", True, {"path": "billing.py", "content": "x=1"})
    assert d.allow is True


def test_blocks_test_file_write():
    d = _hooks().pre_tool_use("write_file", True, {"path": "test_billing.py", "content": "pass"})
    assert d.allow is False
    assert "protected test file" in d.reason


def test_blocks_path_traversal():
    d = _hooks().pre_tool_use("write_file", True, {"path": "../../etc/passwd", "content": "x"})
    assert d.allow is False


def test_blocks_non_writable_extension():
    d = _hooks().pre_tool_use("write_file", True, {"path": "evil.sh", "content": "rm -rf /"})
    assert d.allow is False


def test_redacts_secrets_in_tool_input():
    hooks = _hooks()
    d = hooks.pre_tool_use("read_file", False, {"path": "note", "content": "api_key=ABCDEF1234567890"})
    assert d.redactions >= 1
    assert "REDACTED" in d.tool_input["content"]


def test_read_only_tool_is_allowed():
    assert _hooks().pre_tool_use("grep", False, {"pattern": "x"}).allow is True

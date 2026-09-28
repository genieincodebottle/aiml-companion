"""The Gemini worker seam: mode selection, schema conversion, and key redaction.

These tests do NOT need google-genai installed or a key - the network-facing loop is
exercised live, but the seams around it (worker selection in `harness/worker.py`, the
ToolSpec -> function-declaration conversion, secret redaction) are deterministic.
"""

from __future__ import annotations

import src.config as config
from src.config import Settings
from src.guardrails.hooks import HookManager
from src.harness.base import ToolSpec
from src.harness.gemini_agent import GeminiMigrator, build_function_declarations
from src.harness.stub_agent import StubMigrator
from src.harness.worker import build_worker


def _settings(**overrides) -> Settings:
    base = dict(anthropic_api_key="", gemini_api_key="", force_stub=False)
    base.update(overrides)
    return Settings(**base)


def test_mode_is_stub_without_any_key(monkeypatch):
    monkeypatch.setattr(config, "_genai_installed", lambda: True)
    s = _settings()
    assert s.execution_mode == "stub"
    assert isinstance(build_worker(s), StubMigrator)


def test_gemini_selected_when_key_and_package_present(monkeypatch):
    monkeypatch.setattr(config, "_genai_installed", lambda: True)
    monkeypatch.setattr(config, "_sdk_installed", lambda: False)
    s = _settings(gemini_api_key="AIzaTESTKEY")
    assert s.execution_mode == "live-gemini"
    assert s.use_live_worker
    worker = build_worker(s)
    assert isinstance(worker, GeminiMigrator)
    assert worker.model == s.gemini_model


def test_claude_sdk_outranks_gemini(monkeypatch):
    monkeypatch.setattr(config, "_genai_installed", lambda: True)
    monkeypatch.setattr(config, "_sdk_installed", lambda: True)
    s = _settings(anthropic_api_key="sk-ant-x", gemini_api_key="AIzaTESTKEY")
    assert s.execution_mode == "live-sdk"


def test_force_stub_beats_gemini(monkeypatch):
    monkeypatch.setattr(config, "_genai_installed", lambda: True)
    s = _settings(gemini_api_key="AIzaTESTKEY", force_stub=True)
    assert s.execution_mode == "stub"
    assert isinstance(build_worker(s), StubMigrator)


def test_function_declaration_conversion():
    specs = {
        "grep": ToolSpec(
            name="grep",
            description="Search the repository.",
            handler=lambda *_a, **_k: {},
            input_schema={"pattern": "string", "glob": "string?"},
        ),
        "run_tests": ToolSpec(
            name="run_tests",
            description="Run the tests.",
            handler=lambda *_a, **_k: {},
            input_schema={},
        ),
    }
    decls = {d["name"]: d for d in build_function_declarations(specs)}

    grep = decls["grep"]
    assert grep["parameters"]["properties"] == {"pattern": {"type": "STRING"}, "glob": {"type": "STRING"}}
    assert grep["parameters"]["required"] == ["pattern"]  # optional '?' params excluded

    # A no-arg tool must not carry an empty parameters object.
    assert "parameters" not in decls["run_tests"]


def test_google_api_key_is_redacted():
    text, n = HookManager.redact("using key AIzaSyA1234567890abcdefghijklmnopqrstuv to call gemini")
    assert n == 1
    assert "AIza" not in text

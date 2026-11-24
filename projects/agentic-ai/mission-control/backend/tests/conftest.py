"""Shared fixtures. Every test runs in stub mode against a temp data dir - no key,
no network, fully reproducible."""

from __future__ import annotations

import dataclasses

import pytest

from src.config import get_settings, reset_settings_cache
from src.control_plane import reset_control_plane


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch, tmp_path):
    """Force stub mode for every test.

    Three things, and all three are load-bearing:

    * `MC_FORCE_STUB` is the documented switch.
    * The API keys are DELETED, not just overridden by the switch. If a future
      refactor reintroduces an import-time settings snapshot, the switch is read
      too late to help but a missing key still cannot become a live provider.
      A test suite that can silently bill a real API needs the belt as well as
      the braces.
    * Both singletons are reset. Settings alone are not enough -- the control
      plane binds a provider when it is constructed, so a plane built under a
      live environment stays live however the settings are reset afterwards.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MC_MODE", "offline")
    monkeypatch.delenv("MC_OPERATORS", raising=False)
    monkeypatch.delenv("MC_OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("MC_AUTO_APPROVE", raising=False)
    monkeypatch.setenv("MC_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    reset_control_plane()
    yield
    reset_settings_cache()
    reset_control_plane()


@pytest.fixture
def settings(tmp_path):
    return dataclasses.replace(get_settings(), data_dir=tmp_path)

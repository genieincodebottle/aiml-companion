"""Tests for configuration loader."""

import os
import pytest
from pathlib import Path

# Ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, reset_config, get_agent_config, get_pipeline_config, get_budget_config


@pytest.fixture(autouse=True)
def clean_config():
    """Reset cached config between tests."""
    reset_config()
    yield
    reset_config()


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config()
        assert "model" in config
        assert "agents" in config
        assert "pipeline" in config

    def test_model_defaults(self):
        config = load_config()
        assert config["model"]["provider"] == "google"

    def test_caches_config(self):
        c1 = load_config()
        c2 = load_config()
        assert c1 is c2

    def test_missing_file_uses_defaults(self):
        config = load_config("/nonexistent/path.yaml")
        assert "model" in config
        assert config["model"]["provider"] == "google"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DD_LOG_LEVEL", "DEBUG")
        config = load_config()
        assert config["logging"]["level"] == "DEBUG"

    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("DD_MAX_COST_USD", "1.25")
        config = load_config()
        assert config["budget"]["max_cost_usd"] == 1.25


class TestAgentConfig:
    def test_known_agent(self):
        cfg = get_agent_config("financial_analyst")
        assert "role" in cfg
        assert "temperature" in cfg

    def test_unknown_agent_returns_defaults(self):
        cfg = get_agent_config("nonexistent_agent")
        assert cfg["role"] == "nonexistent_agent"
        assert "temperature" in cfg


class TestPipelineConfig:
    def test_has_required_keys(self):
        cfg = get_pipeline_config()
        assert "max_parallel_agents" in cfg
        assert "max_retries_per_agent" in cfg
        assert "enable_self_correction" in cfg


class TestBudgetConfig:
    def test_has_required_keys(self):
        cfg = get_budget_config()
        assert "max_total_tokens" in cfg
        assert "max_cost_usd" in cfg
        assert cfg["max_total_tokens"] > 0

"""Gaps that were documented and not true, pinned by tests.

No API key or network needed.

Run: python -m pytest tests/test_wiring_gaps.py -v
"""
import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def run_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def _clear_keys(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def test_analyse_refuses_without_any_key(run_module, monkeypatch, capsys):
    _clear_keys(monkeypatch)
    monkeypatch.setattr(run_module, "_main_py", lambda argv: pytest.fail("ran without a key"))
    with pytest.raises(SystemExit) as exc:
        run_module.cmd_analyse(["--company", "X"])
    assert exc.value.code == 2
    assert "GOOGLE_API_KEY" in capsys.readouterr().err


def test_analyse_refuses_a_placeholder_key(run_module, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "your_key_here")
    with pytest.raises(SystemExit):
        run_module.cmd_analyse([])


def test_analyse_accepts_a_real_looking_key(run_module, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyDummyDummyDummy123456")
    seen = []
    monkeypatch.setattr(run_module, "_main_py", lambda argv: seen.append(argv))
    run_module.cmd_analyse(["--company", "X"])
    assert seen == [["--company", "X"]]


def test_run_py_loads_dotenv_before_any_command_runs():
    """A key that lives only in .env used to be reported missing by `analyse`."""
    tree = ast.parse((ROOT / "run.py").read_text(encoding="utf-8"))
    load_line = next(n.lineno for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "load_dotenv")
    first_def = min(n.lineno for n in tree.body if isinstance(n, ast.FunctionDef))
    assert load_line < first_def


def test_dotenv_key_reaches_the_check(tmp_path, monkeypatch):
    """Run the real loader against a .env file: the key must be visible to analyse."""
    from dotenv import load_dotenv

    _clear_keys(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=AIzaSyDummyDummyDummy123456\n", encoding="utf-8")
    load_dotenv(env)
    monkeypatch.syspath_prepend(str(ROOT))
    from main import _has_real_key
    assert _has_real_key("GOOGLE_API_KEY")


def test_unused_config_keys_and_dead_files_are_gone():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert "enable_self_correction" not in cfg["pipeline"]
    assert "max_fact_check_depth" not in cfg["pipeline"]
    assert not (ROOT / "src" / "memory").exists()
    assert not (ROOT / "evaluation" / "judge_prompt.py").exists()


def test_every_pipeline_key_in_the_config_is_read_by_code():
    """Guard against a key that does something in the YAML and nothing in the code."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    for key in cfg["pipeline"]:
        assert f'"{key}"' in source or f"'{key}'" in source, f"pipeline.{key} is read by no code"


def test_eval_companies_carry_no_unscored_expectations():
    from evaluation.run_eval import EVAL_COMPANIES

    assert len(EVAL_COMPANIES) == 3
    for company in EVAL_COMPANIES:
        assert "expected_industry" not in company and "expected_risks" not in company

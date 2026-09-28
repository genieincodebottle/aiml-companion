"""
The test runner's failure paths.

Why these matter more than they look
------------------------------------
`run_tests` produces `tests_passing`, and everything downstream is built on it:
the reviewer's approval, `migration_success_rate`, `clean_migration_rate`, and the
golden promotion gate. It is the single signal the whole eval rests on.

Its happy path was well covered. Its *failure* paths - timeout, missing command,
no tests found - were not (59% line coverage), and those are exactly the paths
where a wrong answer is dangerous. Every one of them must fail CLOSED: if we
cannot demonstrate the tests passed, they did not pass. A timeout that returned
`passed: True` would hand the reviewer a green light for a migration nobody
verified, and the reward-hacking gate in ADR-0003 would be the only thing left
standing.

Run: pytest tests/test_testrunner_fails_closed.py -v
"""
from __future__ import annotations

import sys

import pytest

from src.tools.testrunner import run_tests


def _repo(tmp_path, **files):
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


# === The happy path, so the failures below mean something ===

def test_a_passing_test_reports_passing(tmp_path):
    repo = _repo(tmp_path, **{"test_ok.py": "assert 1 == 1\n"})
    result = run_tests(repo)
    assert result["passed"] is True
    assert result["test_files"] == ["test_ok.py"]


def test_a_failing_test_reports_failing(tmp_path):
    repo = _repo(tmp_path, **{"test_bad.py": "raise SystemExit(1)\n"})
    assert run_tests(repo)["passed"] is False


def test_one_failure_among_several_fails_the_whole_repo(tmp_path):
    """Aggregation must be AND, not OR. A partial green is not green."""
    repo = _repo(tmp_path, **{
        "test_a.py": "assert True\n",
        "test_b.py": "raise SystemExit(1)\n",
        "test_c.py": "assert True\n",
    })
    result = run_tests(repo)
    assert result["passed"] is False
    assert len(result["test_files"]) == 3


# === Fail closed: absence of evidence is not evidence of passing ===

def test_a_repo_with_no_tests_does_not_report_passing(tmp_path):
    """The vacuous-truth trap. `all([])` is True, and an agent that deletes every
    test would look green if this returned the natural aggregation of nothing."""
    result = run_tests(tmp_path)
    assert result["passed"] is False
    assert "no test" in result["output"].lower()
    assert result["test_files"] == []


def test_a_test_that_hangs_is_a_failure_not_a_pass(tmp_path):
    """A timeout means we do not know. Not knowing is not passing."""
    import src.tools.testrunner as tr

    repo = _repo(tmp_path, **{"test_hang.py": "import time\ntime.sleep(30)\n"})
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tr, "_TIMEOUT_S", 1)
    try:
        result = tr.run_tests(repo)
    finally:
        monkey.undo()

    assert result["passed"] is False
    assert "timed out" in result["output"]


def test_a_test_that_crashes_on_import_is_a_failure(tmp_path):
    repo = _repo(tmp_path, **{"test_broken.py": "import a_module_that_does_not_exist\n"})
    assert run_tests(repo)["passed"] is False


# === Explicit command mode ===

def test_an_explicit_command_that_succeeds_reports_passing(tmp_path):
    result = run_tests(tmp_path, command=(sys.executable, "-c", "pass"))
    assert result["passed"] is True


def test_an_explicit_command_that_fails_reports_failing(tmp_path):
    result = run_tests(tmp_path, command=(sys.executable, "-c", "raise SystemExit(2)"))
    assert result["passed"] is False


def test_a_missing_test_command_is_a_failure_not_a_crash(tmp_path):
    """A misconfigured command must fail the repo, not take down the fleet run."""
    result = run_tests(tmp_path, command=("definitely-not-a-real-binary", "--version"))
    assert result["passed"] is False
    assert "not found" in result["output"]


def test_an_explicit_command_that_hangs_is_a_failure(tmp_path):
    import src.tools.testrunner as tr

    monkey = pytest.MonkeyPatch()
    monkey.setattr(tr, "_TIMEOUT_S", 1)
    try:
        result = tr.run_tests(tmp_path, command=(sys.executable, "-c", "import time; time.sleep(30)"))
    finally:
        monkey.undo()

    assert result["passed"] is False
    assert "timed out" in result["output"]


# === Output handling ===

def test_non_ascii_test_output_does_not_crash_the_runner(tmp_path):
    """On Windows, `text=True` decodes a subprocess with cp1252 and raises
    UnicodeDecodeError on pytest's box-drawing characters. The runner pins
    `encoding="utf-8", errors="replace"` for this reason; this pins the reason.

    The child writes UTF-8 bytes to stdout directly rather than calling `print`,
    because a child Python printing non-ASCII to a *pipe* on Windows selects
    cp1252 for its own stdout and dies with UnicodeEncodeError before the runner
    ever sees the output. That is real Windows behaviour and worth knowing, but
    it is a different failure from the decode bug under test here - and it is
    what made the first version of this test fail for the wrong reason.
    """
    payload = "import sys\nsys.stdout.buffer.write('─│ café ✔\\n'.encode('utf-8'))\n"
    repo = _repo(tmp_path, **{"test_unicode.py": payload})

    result = run_tests(repo)          # must not raise UnicodeDecodeError
    assert result["passed"] is True
    assert isinstance(result["output"], str)


def test_the_output_names_which_test_file_failed(tmp_path):
    """A fleet run reports one line per repo; "something failed" is not actionable."""
    repo = _repo(tmp_path, **{
        "test_good.py": "assert True\n",
        "test_evil.py": "raise SystemExit(1)\n",
    })
    output = run_tests(repo)["output"]
    assert "test_evil.py" in output
    assert "FAIL" in output

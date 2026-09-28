"""A clone in a deep Windows folder must still run.

Git's paths below the data dir add about 110 characters. Under a deep checkout
(Documents, OneDrive, a nested workspace) that crosses Windows' path limits and git
fails with "Filename too long", or "Invalid argument" once long paths are on but the
working directory is still too long. Two guards cover it and are tested here.
"""

from pathlib import Path

from src.config import _WINDOWS_DATA_DIR_MAX, _default_data_dir
from src.vcs.gitcmd import _git_env

SHORT = Path("C:/src/aiml-companion/projects/agentic-ai/migration-engineer/backend")
DEEP = Path(
    "C:/Users/firstname.lastname/OneDrive - Contoso Ltd/Documents/GitHub/learning"
    "/aiml-companion/projects/agentic-ai/migration-engineer/backend"
)


def test_short_windows_checkout_keeps_the_data_dir_in_the_repo():
    assert len(str(SHORT / ".work")) <= _WINDOWS_DATA_DIR_MAX
    assert _default_data_dir(SHORT, is_windows=True) == SHORT / ".work"


def test_deep_windows_checkout_moves_the_data_dir_somewhere_short(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/firstname.lastname/AppData/Local")
    chosen = _default_data_dir(DEEP, is_windows=True)
    assert chosen.parent == Path("C:/Users/firstname.lastname/AppData/Local")
    assert chosen.name.startswith("migration-engineer-")
    assert len(str(chosen)) <= _WINDOWS_DATA_DIR_MAX


def test_two_deep_clones_do_not_share_a_data_dir(monkeypatch):
    # Bare remotes are reused between runs, so a shared folder would let one clone
    # test against the other clone's fixtures.
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/me/AppData/Local")
    other = Path(str(DEEP).replace("learning", "learning-copy"))
    assert _default_data_dir(DEEP, is_windows=True) != _default_data_dir(other, is_windows=True)
    assert _default_data_dir(DEEP, is_windows=True) == _default_data_dir(DEEP, is_windows=True)


def test_other_platforms_always_use_the_repo_data_dir():
    assert _default_data_dir(DEEP, is_windows=False) == DEEP / ".work"


def test_git_gets_long_paths_through_the_environment(monkeypatch):
    # The environment, not `-c`, is what reaches the second git process a local
    # push starts to receive the objects.
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    env = _git_env()
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("core.longpaths", "true")


def test_git_env_keeps_config_the_user_already_passed(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.proxy")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "http://proxy.local:8080")
    env = _git_env()
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_0"] == "http.proxy"
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("core.longpaths", "true")


def test_git_env_leaves_a_malformed_count_for_git_to_report(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "not-a-number")
    env = _git_env()
    assert env["GIT_CONFIG_COUNT"] == "not-a-number"
    assert "GIT_CONFIG_KEY_0" not in env or env.get("GIT_CONFIG_KEY_0") != "core.longpaths"

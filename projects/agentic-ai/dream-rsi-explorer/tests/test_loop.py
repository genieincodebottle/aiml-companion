import random

import pytest

from src.agents import GeminiAgent
from src.develop import GeminiDeveloper, dream
from src.explore import explore
from src.gemini import find_api_key
from src.loop import Settings, fresh_check, run_arm
from src.policy import BOUNDS, HAND_WRITTEN, PolicySpec
from src.replay import score_on_pool
from src.task import evaluate
from src.tree import DiscoveryTree
from tests.fakes import FakeGeminiClient, random_layout


def agent(client=None):
    return GeminiAgent(client or FakeGeminiClient(), "fake")


def test_hand_written_search_makes_24_calls():
    client = FakeGeminiClient()
    result = explore(HAND_WRITTEN, agent(client), seed=1, max_calls=100)
    assert result.calls == client.calls == 24


def test_explore_respects_the_call_cap():
    result = explore(HAND_WRITTEN, agent(), seed=1, max_calls=7)
    assert result.calls <= 7 and result.tree.attempts == result.calls


def test_explore_tree_is_stable_under_parallel_calls():
    a = explore(HAND_WRITTEN, agent(), seed=4, max_calls=50, workers=8)
    b = explore(HAND_WRITTEN, agent(), seed=4, max_calls=50, workers=1)
    assert a.tree.to_dict() == b.tree.to_dict()


def test_failed_calls_are_recorded_not_fatal():
    result = explore(HAND_WRITTEN, agent(FakeGeminiClient(fail_every=3)), seed=2, max_calls=24)
    failed = [n for n in result.tree.nodes.values() if n.diagnostics == "agent call failed"]
    assert failed and all(n.score == 0.0 for n in failed)
    assert result.calls == 24


def test_agent_prompt_carries_parent_and_harness_fits_radii():
    client = FakeGeminiClient()
    tree = DiscoveryTree()
    node = tree.nodes[tree.add(0, 1.23, random_layout(random.Random(1)), "valid, smallest circle 4", 0)]
    proposal = GeminiAgent(client, "fake").propose(node, random.Random(2))
    assert "1.23" in client.prompts[-1] and "smallest circle 4" in client.prompts[-1]
    assert evaluate(proposal).valid          # radii come from the harness, never from the model


def developer(client):
    return GeminiDeveloper(client, "fake", beta1=0.004, beta2=0.002)


def pool(n=2):
    return [explore(HAND_WRITTEN, agent(), seed=s, max_calls=24).tree for s in range(n)]


def test_dreaming_never_calls_the_discovery_agent():
    discovery = FakeGeminiClient()
    trees = [explore(HAND_WRITTEN, agent(discovery), seed=s, max_calls=24).tree for s in range(2)]
    before = discovery.calls
    dev_client = FakeGeminiClient()
    result = dream(HAND_WRITTEN, trees, developer(dev_client), versions=4, beta1=0.004, beta2=0.002)
    assert discovery.calls == before
    assert dev_client.calls == 4
    assert result.chosen == max(result.tried, key=lambda t: t.value).spec


def test_developer_prompt_carries_the_replay_statistics():
    client = FakeGeminiClient()
    trees = pool()
    tried = [score_on_pool(HAND_WRITTEN, trees, beta1=0.004, beta2=0.002)]
    developer(client).propose(tried[0], tried)
    prompt = client.prompts[-1]
    assert "share_of_calls_after_last_gain" in prompt and "0.004" in prompt


def test_developer_output_is_clamped():
    wild = {"reason": "go wide", "width": 50, "attempts_per_node": 0, "restarts": 9, "patience": 2,
            "on_stall": "widen", "stall_boost": 2, "max_stalled_rounds": 3, "max_rounds": 99}
    tried = [score_on_pool(HAND_WRITTEN, pool(1), beta1=0.004, beta2=0.002)]
    spec, why = developer(FakeGeminiClient(policy_reply=wild)).propose(tried[0], tried)
    assert spec.width == BOUNDS["width"][1] and spec.attempts_per_node == 1
    assert spec.max_rounds == BOUNDS["max_rounds"][1] and why == "go wide"


def test_dream_keeps_the_current_policy_if_nothing_beats_it():
    worse = {"reason": "worse", "width": 1, "attempts_per_node": 1, "restarts": 0, "patience": 6,
             "on_stall": "stop", "stall_boost": 0, "max_stalled_rounds": 6, "max_rounds": 2}
    result = dream(HAND_WRITTEN, pool(), developer(FakeGeminiClient(policy_reply=worse)),
                   versions=3, beta1=0.0, beta2=0.0)
    assert result.chosen == HAND_WRITTEN


def test_a_failing_developer_does_not_break_the_loop():
    result = dream(HAND_WRITTEN, pool(1), developer(FakeGeminiClient(fail_every=1)),
                   versions=2, beta1=0.004, beta2=0.002)
    assert result.chosen == HAND_WRITTEN
    assert any("developer failed" in n for n in result.notes)


def test_arms_share_round_zero_and_only_dream_changes_policy():
    s = Settings(rounds=3, versions=2, max_calls=24, fresh_seeds=2, workers=4)
    round0 = explore(HAND_WRITTEN, agent(), seed=s.seed * 1000, max_calls=s.max_calls)
    fixed = run_arm("fixed", HAND_WRITTEN, agent(), s, first_round=round0)
    dreamt = run_arm("dream", HAND_WRITTEN, agent(), s, developer=developer(FakeGeminiClient()), first_round=round0)
    assert fixed.rounds[0].best_score == dreamt.rounds[0].best_score
    assert all(r.policy == HAND_WRITTEN for r in fixed.rounds) and fixed.final_policy == HAND_WRITTEN
    assert all(r.chosen_replay_value is not None for r in dreamt.rounds)


def test_fresh_check_reports_value_with_cost():
    s = Settings(max_calls=24, fresh_seeds=3, workers=4)
    check = fresh_check(HAND_WRITTEN, agent(), s)
    assert check.mean_calls == 24
    assert check.mean_value == pytest.approx(check.mean_best - 0.004 * 24 + 0.002 * 24 / 6)


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    assert find_api_key() == "from-environment"


@pytest.mark.parametrize(
    "raw",
    [
        "GEMINI_API_KEY=abc123\n".encode("utf-8"),
        "﻿GEMINI_API_KEY=abc123\r\n".encode("utf-8"),        # Notepad with a byte-order mark
        "GEMINI_API_KEY=abc123\r\n".encode("utf-16"),              # PowerShell `echo ... > .env`
        'export GEMINI_API_KEY="abc123"\n'.encode("utf-8"),
    ],
)
def test_env_file_is_read_in_common_encodings(tmp_path, monkeypatch, raw):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env").write_bytes(raw)
    nested = tmp_path / "src"
    nested.mkdir()
    assert find_api_key(start=nested) == "abc123"      # found by walking up from a subfolder

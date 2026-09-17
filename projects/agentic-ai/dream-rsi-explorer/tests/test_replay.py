import pytest

from src.agents import GeminiAgent
from src.explore import explore
from src.policy import PolicySpec
from src.replay import replay, score_on_pool
from src.tree import ROOT, DiscoveryTree
from tests.fakes import FakeGeminiClient


def small_tree() -> DiscoveryTree:
    """root -> a(1.0) -> c(1.4)
       root -> b(1.2)"""
    tree = DiscoveryTree()
    a = tree.add(ROOT, 1.0, None, "", 0)
    tree.add(ROOT, 1.2, None, "", 0)
    tree.add(a, 1.4, None, "", 1)
    return tree


def test_replay_only_reveals_recorded_nodes():
    tree = small_tree()
    spec = PolicySpec(width=2, attempts_per_node=4, restarts=4, patience=6, max_rounds=5)
    result = replay(spec, tree, beta1=0.0, beta2=0.0)
    assert result.revealed == 3            # everything recorded, nothing more
    assert result.best_score == pytest.approx(1.4)
    assert result.empty_picks > 0          # it asked for more than was recorded


def test_replay_score_matches_the_paper_formula():
    tree = small_tree()
    spec = PolicySpec(width=1, attempts_per_node=1, restarts=2, patience=6, max_rounds=2)
    r = replay(spec, tree, beta1=0.01, beta2=0.02)
    expected = r.best_score - 0.01 * r.revealed + 0.02 * r.revealed / max(1, r.rounds)
    assert r.value == pytest.approx(expected)


def test_children_are_revealed_in_recorded_order():
    tree = DiscoveryTree()
    first = tree.add(ROOT, 0.5, None, "", 0)
    tree.add(ROOT, 0.9, None, "", 0)
    one_restart = PolicySpec(width=1, attempts_per_node=1, restarts=1, patience=6, max_rounds=1)
    r = replay(one_restart, tree, beta1=0.0, beta2=0.0)
    assert r.revealed == 1 and r.best_score == tree.nodes[first].score


def test_replay_never_calls_the_model():
    client = FakeGeminiClient()
    live = explore(PolicySpec(max_rounds=3), GeminiAgent(client, "fake"), seed=1, max_calls=50)
    before = client.calls
    replay(PolicySpec(width=3, attempts_per_node=3, max_rounds=10), live.tree, beta1=0.0, beta2=0.0)
    assert client.calls == before


def test_replay_of_a_full_recording_matches_the_live_run():
    """Replaying the policy that produced a tree reproduces that run exactly."""
    spec = PolicySpec(width=2, attempts_per_node=2, restarts=1, patience=3, on_stall="stop", max_rounds=5)
    live = explore(spec, GeminiAgent(FakeGeminiClient(), "fake"), seed=11, max_calls=1000)
    r = replay(spec, live.tree, beta1=0.0, beta2=0.0)
    assert r.revealed == live.calls
    assert r.best_score == pytest.approx(live.best_score)
    assert r.best_by_round == pytest.approx(live.best_by_round)


def test_pool_average():
    pool = [small_tree(), small_tree()]
    spec = PolicySpec(width=1, attempts_per_node=1, restarts=1, max_rounds=3)
    single = replay(spec, pool[0], beta1=0.001, beta2=0.0)
    assert score_on_pool(spec, pool, beta1=0.001, beta2=0.0).value == pytest.approx(single.value)

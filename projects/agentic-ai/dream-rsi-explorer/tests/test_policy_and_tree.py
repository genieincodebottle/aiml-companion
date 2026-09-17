from src.policy import HAND_WRITTEN, PolicySpec, View, decide, stalled_rounds
from src.tree import ROOT, DiscoveryTree


def test_tree_leaves_best_path_and_round_trip(tmp_path):
    tree = DiscoveryTree()
    a = tree.add(ROOT, 1.0, None, "a", 0)
    b = tree.add(ROOT, 2.0, None, "b", 0)
    c = tree.add(a, 3.0, None, "c", 1)
    assert sorted(tree.leaves()) == [b, c]
    assert tree.best().id == c
    assert tree.path(c) == "0.0" and tree.path(b) == "1"
    tree.save(tmp_path / "t.json")
    loaded = DiscoveryTree.load(tmp_path / "t.json")
    assert loaded.to_dict() == tree.to_dict()


def test_decisions_only_pick_root_or_leaves():
    view = View(round=0, leaves=[(5, 1.2), (7, 1.5), (9, 1.1)], best_by_round=[])
    decision = decide(PolicySpec(width=2, attempts_per_node=3, restarts=1), view)
    assert set(decision) <= {ROOT, 5, 7, 9}
    assert decision == {7: 3, 5: 3, ROOT: 1}   # best first


def test_empty_tree_always_starts_from_root():
    decision = decide(PolicySpec(restarts=0), View(round=0, leaves=[], best_by_round=[]))
    assert decision == {ROOT: 1}


def test_stall_detection_and_reactions():
    assert stalled_rounds([1.0, 1.1, 1.1, 1.1]) == 2
    assert stalled_rounds([1.0, 1.1]) == 0
    stalled = View(round=5, leaves=[(1, 1.0), (2, 0.9)], best_by_round=[1.0, 1.0, 1.0, 1.0])

    assert decide(PolicySpec(patience=2, on_stall="stop"), stalled) == {}
    widen = decide(PolicySpec(width=1, patience=2, on_stall="widen", stall_boost=1, restarts=0), stalled)
    assert set(widen) == {1, 2}
    restart = decide(PolicySpec(width=1, patience=2, on_stall="restart", stall_boost=2, restarts=1), stalled)
    assert restart[ROOT] == 3


def test_max_rounds_stops_the_search():
    assert decide(HAND_WRITTEN, View(round=HAND_WRITTEN.max_rounds, leaves=[(1, 1.0)], best_by_round=[])) == {}


def test_clamp_keeps_every_knob_in_bounds():
    wild = PolicySpec.from_dict({"width": 99, "restarts": -5, "on_stall": "explode", "max_rounds": "3", "junk": 1})
    assert wild.width == 6 and wild.restarts == 0 and wild.on_stall == "stop" and wild.max_rounds == 3
    assert HAND_WRITTEN.clamp() == HAND_WRITTEN

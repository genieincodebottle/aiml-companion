"""The file log behaves like Kafka where it matters to the sinks."""
import pytest

from log_replay.log.base import Consumer, partition_for
from log_replay.log.filelog import FileLog


@pytest.fixture
def log(tmp_path):
    lg = FileLog(tmp_path, "t", 3, create=True)
    for i in range(30):
        key = str(i % 7)
        lg.append(partition_for(key, 3), key, {"n": i})
    lg.flush()
    return FileLog(tmp_path, "t", 3)


def test_offsets_are_per_partition_and_dense(log):
    for p in range(3):
        assert [r.offset for r in log.read(p, 0, 1000)] == list(range(log.end_offset(p)))


def test_same_key_keeps_its_order_inside_one_partition(log):
    for p in range(3):
        by_key = {}
        for r in log.read(p, 0, 1000):
            by_key.setdefault(r.key, []).append(r.value["n"])
        assert all(v == sorted(v) for v in by_key.values())


def test_poll_advances_position_but_commit_is_what_survives_a_restart(log):
    c = Consumer(log, "g")
    first = c.poll(10)
    assert len(first) == 10 and sum(c.position.values()) == 10
    assert Consumer(log, "g").position == {0: 0, 1: 0, 2: 0}  # nothing committed yet
    c.commit()
    assert Consumer(log, "g").position == c.position


def test_committed_offsets_are_stored_apart_from_the_data(log, tmp_path):
    Consumer(log, "g").commit()
    assert (tmp_path / "__consumer_offsets" / "g.jsonl").exists()
    assert not any("g" == p.name for p in (tmp_path / "t").iterdir())


def test_groups_do_not_share_offsets(log):
    a = Consumer(log, "a")
    a.poll(5)
    a.commit()
    assert Consumer(log, "b").position == {0: 0, 1: 0, 2: 0}


def test_commits_survive_reopening_the_log(log, tmp_path):
    a = Consumer(log, "g")
    a.poll(12)
    a.commit()
    log.flush()
    assert Consumer(FileLog(tmp_path, "t", 3), "g").position == a.position


def test_seek_and_reset_rewind_the_consumer(log):
    c = Consumer(log, "g")
    c.poll(20)
    c.commit()
    c.seek(0, 0)
    assert c.position[0] == 0
    log.reset_group("g")
    assert Consumer(log, "g").position == {0: 0, 1: 0, 2: 0}


def test_poll_returns_every_record_exactly_once_then_empty(log):
    c, got = Consumer(log, "g"), []
    while True:
        batch = c.poll(7)
        if not batch:
            break
        got += [(r.partition, r.offset) for r in batch]
    assert len(got) == 30 and len(set(got)) == 30

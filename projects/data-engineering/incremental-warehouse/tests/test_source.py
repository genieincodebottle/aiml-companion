"""The simulated source, and the failures planted in it."""
import hashlib
import json
from datetime import datetime

from warehouse.source.shop import Shop
from warehouse.source.world import generate_world


def _digest(days):
    return hashlib.sha256(json.dumps(days, sort_keys=True).encode()).hexdigest()


def test_world_is_deterministic(cfg, days):
    again, _ = generate_world(cfg)
    assert _digest(days) == _digest(again)


def test_planted_failures_have_the_documented_sizes(key):
    assert len(key.orders) == 1789
    assert len(key.late_events) == 111
    assert sum(1 for e in key.late_events if e["table"] == "orders") == 103
    assert sum(1 for e in key.late_events if e["table"] == "customers") == 8
    assert len(key.deletes) == 27


def test_a_late_commit_is_stamped_before_midnight_but_applied_that_day(cfg, days, key):
    inside = 0
    for ev in key.late_events:
        visible = datetime.strptime(days[ev["day"] - 1]["date"], "%Y-%m-%d")
        stamp = datetime.strptime(ev["ts"], "%Y-%m-%d %H:%M:%S")
        gap = (visible - stamp).total_seconds() / 60
        assert 0 < gap <= cfg["source"]["late_max_minutes"]
        inside += 1
    assert inside == len(key.late_events)


def test_a_late_order_is_absent_the_day_before_it_becomes_visible(days, key):
    late = next(e for e in key.late_events if e["table"] == "orders" and e["day"] > 2)
    shop = Shop(":memory:", days)
    shop.advance_to(late["day"] - 1)
    seen = shop.conn.execute("SELECT count(*) FROM orders WHERE order_id = ? AND updated_at = ?",
                             (late["key"], late["ts"])).fetchone()[0]
    assert seen == 0
    shop.advance_to(late["day"])
    seen = shop.conn.execute("SELECT count(*) FROM orders WHERE order_id = ? AND updated_at = ?",
                             (late["key"], late["ts"])).fetchone()[0]
    assert seen == 1


def test_a_hard_deleted_order_leaves_no_row_behind(days, key):
    gone = key.deletes[0]
    shop = Shop(":memory:", days)
    shop.advance_to(gone["day"] - 1)
    assert shop.conn.execute("SELECT count(*) FROM orders WHERE order_id = ?",
                             (gone["order_id"],)).fetchone()[0] == 1
    shop.advance_to(gone["day"])
    assert shop.conn.execute("SELECT count(*) FROM orders WHERE order_id = ?",
                             (gone["order_id"],)).fetchone()[0] == 0
    assert shop.conn.execute("SELECT count(*) FROM order_lines WHERE order_id = ?",
                             (gone["order_id"],)).fetchone()[0] == 0


def test_replay_gives_the_same_database_as_running_forward(days):
    forward = Shop(":memory:", days)
    forward.advance_to(12)
    replayed = Shop(":memory:", days)
    replayed.advance_to(20)
    replayed.advance_to(12)
    dump = lambda s: list(s.conn.iterdump())
    assert dump(forward) == dump(replayed)


def test_end_of_month_row_counts(days):
    shop = Shop(":memory:", days)
    shop.advance_to(30)
    assert shop.counts() == {"customers": 455, "orders": 1762, "order_lines": 4443}

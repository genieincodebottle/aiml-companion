"""Pipeline behaviour that does not need the answer key."""
import pytest

from warehouse import transform
from warehouse.harness import Scenario
from warehouse.options import PipelineOptions


def test_every_daily_run_succeeds(clean):
    assert all(r.ok for r in clean.runs)
    assert len(clean.runs) == 30


def test_memory_store_and_parquet_store_publish_identical_tables(clean, clean_memory):
    assert clean.wh.checksums() == clean_memory.wh.checksums()


def test_incremental_fact_equals_a_full_recompute_after_every_day(cfg, days, opts):
    """The touched-partition logic must never leave a stale partition behind."""
    sc = Scenario(cfg, days, opts)
    for day in range(1, cfg["source"]["days"] + 1):
        assert sc.run_day(day).ok
        conn = sc.wh.conn
        for fact in ("fct_order", "fct_order_line"):
            full = transform.render(fact, cfg, opts)
            diff = conn.execute(
                f"SELECT count(*) FROM (({full}) EXCEPT SELECT * FROM published.{fact}) "
                f"UNION ALL SELECT count(*) FROM "
                f"((SELECT * FROM published.{fact}) EXCEPT ({full}))").fetchall()
            assert diff == [(0,), (0,)], f"{fact} differs from a full rebuild on day {day}"


def test_a_run_rebuilds_old_partitions_not_just_its_own_date(clean):
    last = clean.runs[-1]
    touched = last.xcom["plan_partitions"]
    assert len(touched) == 12
    assert min(touched) < clean.date_of(30).isoformat()


def test_raw_partition_is_replaced_in_place_on_rerun(cfg, days, opts, tmp_path):
    sc = Scenario(cfg, days, opts, root=tmp_path / "r")
    for d in range(1, 6):
        sc.run_day(d)
    path = sc.store.file("orders", sc.date_of(5))
    before = path.read_bytes()
    sc.run_day(5)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["part.parquet"]


def test_publish_is_atomic(cfg, days, opts):
    sc = Scenario(cfg, days, opts)
    for d in range(1, 4):
        sc.run_day(d)
    wh = sc.wh
    before = wh.checksums()
    wh.reset_build()
    wh.conn.execute("DELETE FROM build.fct_order")
    wh.conn.execute("DROP TABLE build.fct_order_line")      # the third swap will fail
    with pytest.raises(Exception):
        wh.publish("test#1", sc.date_of(3))
    assert wh.checksums() == before


def test_transient_failure_is_retried_and_recorded(cfg, days, opts):
    sc = Scenario(cfg, days, opts)
    run = sc.run_day(1, faults={"extract_orders": 2})
    assert run.ok
    row = next(r for r in sc.wh.runs(run.run_id) if r[2] == "extract_orders")
    assert row[3:5] == ("success", 3)


def test_transient_failure_that_never_clears_stops_the_run(cfg, days, opts):
    sc = Scenario(cfg, days, opts)
    run = sc.run_day(1, faults={"extract_orders": 99})
    assert not run.ok
    assert run.status("extract_orders") == "failed"
    assert run.status("publish") == "skipped"
    assert not sc.wh.exists("published", "fct_order")


def test_options_reject_unknown_values(cfg):
    with pytest.raises(ValueError):
        PipelineOptions.from_cfg(cfg, load_mode="merge")
    with pytest.raises(ValueError):
        PipelineOptions.from_cfg(cfg, clock="tomorrow")

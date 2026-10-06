"""Each check fires on the mutation it exists for, and stays quiet on clean data."""
from datetime import date

import pyarrow as pa

from warehouse import checks

C = {"volume_history_days": 7, "volume_min_history": 3, "volume_min_ratio": 0.5,
     "volume_max_ratio": 3.0}


def test_contract_passes_when_columns_match():
    t = pa.table({"a": [1], "b": [2]})
    assert checks.check_contract("t", t, ["a", "b"]).passed


def test_contract_fires_on_a_renamed_column():
    t = pa.table({"a": [1], "b_new": [2]})
    r = checks.check_contract("t", t, ["a", "b"])
    assert not r.passed and "missing=['b']" in r.detail and "unexpected=['b_new']" in r.detail


def test_null_check_fires_on_a_spike_and_ignores_a_clean_column():
    clean = pa.table({"k": list(range(100))})
    spike = pa.table({"k": [None] * 30 + list(range(70))})
    assert checks.check_nulls("t", clean, ["k"], 0.01).passed
    assert not checks.check_nulls("t", spike, ["k"], 0.01).passed


def test_volume_fires_on_a_drop_and_on_a_surge_but_waits_for_history():
    history = {date(2025, 1, d): 200 for d in range(1, 6)}
    assert checks.check_volume(190, history, C).passed
    assert not checks.check_volume(60, history, C).passed
    assert not checks.check_volume(900, history, C).passed
    assert checks.check_volume(5, {date(2025, 1, 1): 200}, C).passed  # not enough history


def test_freshness_fires_when_the_newest_row_is_old():
    run_date = date(2025, 6, 1)
    fresh = pa.table({"updated_at": ["2025-06-01 21:00:00", "2025-06-01 23:30:00"]})
    stale = pa.table({"updated_at": ["2025-06-01 02:00:00"]})
    assert checks.check_freshness(fresh, run_date, 6).passed
    assert not checks.check_freshness(stale, run_date, 6).passed
    assert not checks.check_freshness(pa.table({"updated_at": pa.array([], pa.string())}),
                                      run_date, 6).passed


def test_check_marts_is_clean_on_a_correct_build(clean):
    wh, shop = clean.wh, clean.shop
    wh.reset_build()
    results = checks.check_marts(wh.conn, shop.conn, clean.cfg)
    assert results and all(r.passed for r in results)


def test_check_marts_fires_on_each_planted_corruption(clean):
    wh, shop, cfg = clean.wh, clean.shop, clean.cfg

    def fired(mutation):
        wh.reset_build()
        wh.conn.execute(mutation)
        return {r.name for r in checks.check_marts(wh.conn, shop.conn, cfg) if not r.passed}

    assert "grain_fct_order" in fired(
        "INSERT INTO build.fct_order SELECT * FROM build.fct_order LIMIT 1")
    assert "grain_fct_order_line" in fired(
        "INSERT INTO build.fct_order_line SELECT * FROM build.fct_order_line LIMIT 1")
    assert "grain_dim_customer" in fired(
        "INSERT INTO build.dim_customer_scd2 SELECT * FROM build.dim_customer_scd2 LIMIT 1")
    assert "ri_fct_order_customer" in fired("UPDATE build.fct_order SET customer_sk = NULL "
                                            "WHERE order_id = 1")
    assert "lines_sum_to_order_total" in fired("UPDATE build.fct_order SET order_total_cents = 1 "
                                               "WHERE order_id = 1")
    drop = fired("DELETE FROM build.fct_order WHERE order_id = 1")
    assert {"reconcile_order_count", "reconcile_by_date"} <= drop
    revenue = fired("UPDATE build.fct_order SET revenue_cents = revenue_cents + 1 "
                    "WHERE revenue_cents > 0 AND order_id = (SELECT min(order_id) FROM "
                    "build.fct_order WHERE revenue_cents > 0)")
    assert {"reconcile_revenue", "reconcile_by_date"} <= revenue

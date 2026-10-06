"""Scores a warehouse against the answer key. The only module that reads it.

Everything here compares what the pipeline published with what the shop's data
really was. The pipeline never imports this module, and a test checks that.
"""


def _truth_rows(key, day):
    return {(d, r): c for (d, r), c in key.revenue_as_of(day).items()}


def score_revenue(key, published_rows, day):
    """Compare published revenue by (order_date, region) with the true state after `day`."""
    truth = _truth_rows(key, day)
    got = {(d, r): c for d, r, c in published_rows}
    cells = set(truth) | set(got)
    wrong = [c for c in cells if truth.get(c, 0) != got.get(c, 0)]
    return {
        "truth_cents": sum(truth.values()),
        "published_cents": sum(got.values()),
        "cells": len(cells),
        "cells_wrong": len(wrong),
        "abs_error_cents": sum(abs(truth.get(c, 0) - got.get(c, 0)) for c in cells),
        "exact": not wrong,
    }


def score_every_day(key, published_by_day):
    """Score the revenue published after each run against the truth on that day."""
    per_day = {day: score_revenue(key, rows, day) for day, rows in sorted(published_by_day.items())}
    wrong = [d for d, s in per_day.items() if not s["exact"]]
    worst = max(per_day, key=lambda d: (per_day[d]["abs_error_cents"], -d))
    return {
        "abs_error_cents_by_day": [per_day[d]["abs_error_cents"] for d in per_day],
        "total_error_cents_by_day": [per_day[d]["published_cents"] - per_day[d]["truth_cents"]
                                     for d in per_day],
        "days": len(per_day),
        "days_exact": len(per_day) - len(wrong),
        "first_wrong_day": wrong[0] if wrong else None,
        "worst_day": worst,
        "worst_abs_error_cents": per_day[worst]["abs_error_cents"],
    }


def score_orders(key, wh, day):
    """Order-level comparison of published.fct_order with the true orders after `day`."""
    truth = key.orders_as_of(day)
    rows, fct_rows = {}, 0
    if wh.exists("published", "fct_order"):
        fetched = wh.conn.execute(
            "SELECT order_id, status, region, revenue_cents FROM published.fct_order").fetchall()
        fct_rows = len(fetched)
        rows = {oid: (status, region, rev) for oid, status, region, rev in fetched}
    deleted = {oid for oid, d in key.deleted_on.items() if d <= day}
    return {
        "orders_true": len(truth),
        "fct_rows": fct_rows,
        "orders_missing": len(set(truth) - set(rows)),
        "deleted_still_published": len(deleted & set(rows)),
        "status_stale": sum(1 for oid, t in truth.items()
                            if oid in rows and rows[oid][0] != t["status"]),
        "region_wrong": sum(1 for oid, t in truth.items()
                            if oid in rows and rows[oid][1] != t["region"]),
        "duplicate_rows": fct_rows - len(rows),
    }


def score_dimension(key, wh, day):
    """The published dimension should hold exactly the true (customer, region, start) rows."""
    truth = key.dim_rows_as_of(day)
    got = set()
    if wh.exists("published", "dim_customer_scd2"):
        got = {(c, r, str(v)) for c, r, v in wh.conn.execute(
            "SELECT customer_id, region, valid_from FROM published.dim_customer_scd2").fetchall()}
    return {"truth_rows": len(truth), "published_rows": len(got), "exact": truth == got}


def late_commit_coverage(key, wh, store):
    """How many planted late commits have their version in the raw lake."""
    store.register(wh.conn)
    landed = {"orders": set(), "customers": set()}
    for table, id_col in (("orders", "order_id"), ("customers", "customer_id")):
        if store.partitions(table):
            landed[table] = set(wh.conn.execute(
                f"SELECT {id_col}, updated_at FROM raw_{table}").fetchall())
    planted = key.late_events
    found = sum(1 for e in planted if (e["key"], e["ts"]) in landed[e["table"]])
    return {"planted": len(planted), "landed": found, "missed": len(planted) - found}


def planted_summary(key):
    """The sizes of the planted failures, for the report."""
    by_table = {}
    for e in key.late_events:
        by_table[e["table"]] = by_table.get(e["table"], 0) + 1
    deleted = key.deletes
    return {
        "orders_total": len(key.orders),
        "late_commits": len(key.late_events),
        "late_commits_orders": by_table.get("orders", 0),
        "late_commits_customers": by_table.get("customers", 0),
        "hard_deletes": len(deleted),
    }

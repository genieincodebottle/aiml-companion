"""Data checks. They are pipeline stages, so a failure stops the run.

`check_raw` looks at what was just landed, before any model reads it.
`check_marts` looks at the built tables, before anything is published.

Both return a list of `CheckResult`. The pipeline decides what a failure means.
In `gate` mode it stops the run. In `report` mode (the naive variant) it only
writes the result and the run carries on to publish.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median


class CheckFailed(Exception):
    """Raised by the pipeline when a gating check fails. Not retried."""


@dataclass
class CheckResult:
    stage: str
    name: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# check_raw
# ---------------------------------------------------------------------------
def check_contract(table_name, arrow_table, expected):
    """Column names must match the contract exactly, in both directions."""
    got = set(arrow_table.column_names)
    missing, extra = sorted(set(expected) - got), sorted(got - set(expected))
    ok = not missing and not extra
    detail = "" if ok else f"missing={missing} unexpected={extra}"
    return CheckResult("check_raw", f"contract_{table_name}", ok, detail)


def check_nulls(table_name, arrow_table, required, max_rate):
    """Required columns must stay under the null-rate ceiling."""
    bad = {}
    for col in required:
        if col in arrow_table.column_names and arrow_table.num_rows:
            rate = arrow_table[col].null_count / arrow_table.num_rows
            if rate > max_rate:
                bad[col] = round(rate, 3)
    return CheckResult("check_raw", f"nulls_{table_name}", not bad,
                       "" if not bad else f"null rate above {max_rate}: {bad}")


def check_volume(rows_today, previous_counts, c):
    """Rows landed today against the median of the trailing partitions."""
    history = [n for _, n in sorted(previous_counts.items())][-c["volume_history_days"]:]
    if len(history) < c["volume_min_history"]:
        return CheckResult("check_raw", "volume", True, "skipped, not enough history")
    base = median(history)
    ratio = rows_today / base
    ok = c["volume_min_ratio"] <= ratio <= c["volume_max_ratio"]
    return CheckResult("check_raw", "volume", ok,
                       f"rows={rows_today} trailing_median={base:g} ratio={ratio:.2f}")


def check_freshness(arrow_table, run_date, hours):
    """The newest updated_at must be close to the end of the logical day."""
    if arrow_table.num_rows == 0:
        return CheckResult("check_raw", "freshness", False, "no rows")
    newest = datetime.strptime(max(arrow_table["updated_at"].to_pylist()), "%Y-%m-%d %H:%M:%S")
    end = datetime(run_date.year, run_date.month, run_date.day) + timedelta(days=1)
    lag = (end - newest).total_seconds() / 3600
    return CheckResult("check_raw", "freshness", lag <= hours, f"newest_lag_hours={lag:.2f}")


def check_raw(store, cfg, run_date, partition_date):
    c = cfg["checks"]
    results = []
    for name in ("orders", "order_lines", "customers"):
        tbl = store.read(name, partition_date)
        results.append(check_contract(name, tbl, c["contract"][name]))
        results.append(check_nulls(name, tbl, c["required_not_null"][name], c["max_null_rate"]))
    vol = c["volume_table"]
    counts = store.row_counts(vol)
    today = counts.pop(partition_date)
    earlier = {d: n for d, n in counts.items() if d < partition_date}
    results.append(check_volume(today, earlier, c))
    results.append(check_freshness(store.read("orders", partition_date), run_date,
                                   c["freshness_hours"]))
    return results


# ---------------------------------------------------------------------------
# check_marts
# ---------------------------------------------------------------------------
def _scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


def source_control_totals(source_conn, statuses):
    """Counts and revenue per created date, straight from the OLTP source."""
    marks = ", ".join("?" for _ in statuses)
    rows = source_conn.execute(
        "SELECT substr(o.created_at, 1, 10), count(DISTINCT o.order_id), "
        f"coalesce(sum(CASE WHEN o.status IN ({marks}) "
        "THEN l.quantity * l.unit_price_cents ELSE 0 END), 0) "
        "FROM orders o LEFT JOIN order_lines l USING (order_id) "
        "GROUP BY 1 ORDER BY 1", list(statuses)).fetchall()
    return {d: (n, int(rev)) for d, n, rev in rows}


def check_marts(conn, source_conn, cfg):
    r = []

    def add(name, bad, detail=""):
        r.append(CheckResult("check_marts", name, bad == 0, detail if bad else ""))

    add("grain_fct_order", _scalar(conn,
        "SELECT count(*) - count(DISTINCT order_id) FROM build.fct_order"),
        "duplicate order_id rows in fct_order")
    add("grain_fct_order_line", _scalar(conn,
        "SELECT count(*) - count(DISTINCT (order_id, line_no)) FROM build.fct_order_line"),
        "duplicate (order_id, line_no) rows in fct_order_line")
    add("grain_dim_customer", _scalar(conn,
        "SELECT (SELECT count(*) - count(DISTINCT (customer_id, valid_from)) "
        "        FROM build.dim_customer_scd2) "
        "     + (SELECT count(*) FROM (SELECT customer_id FROM build.dim_customer_scd2 "
        "         GROUP BY 1 HAVING sum(CASE WHEN is_current THEN 1 ELSE 0 END) <> 1))"),
        "a customer has duplicate versions or not exactly one current row")
    add("ri_fct_order_customer", _scalar(conn,
        "SELECT count(*) FROM build.fct_order f LEFT JOIN build.dim_customer_scd2 d "
        "USING (customer_sk) WHERE f.customer_sk IS NULL OR d.customer_sk IS NULL"),
        "orders with no matching customer dimension row")
    add("lines_sum_to_order_total", _scalar(conn,
        "SELECT count(*) FROM build.fct_order f LEFT JOIN "
        "(SELECT order_id, sum(line_total_cents) AS t FROM build.fct_order_line GROUP BY 1) l "
        "USING (order_id) WHERE f.order_total_cents <> coalesce(l.t, 0)"),
        "orders whose total differs from the sum of their lines")

    # Reconciliation against the source is the check that sees what extraction missed.
    try:
        src = source_control_totals(source_conn, cfg["revenue_statuses"])
    except sqlite3.Error as exc:
        r.append(CheckResult("check_marts", "reconcile_source", False,
                             f"source query failed: {exc}"))
        return r
    wh = {d: (int(n), int(rev)) for d, n, rev in conn.execute(
        "SELECT CAST(order_date AS VARCHAR), count(*), sum(revenue_cents) "
        "FROM build.fct_order GROUP BY 1 ORDER BY 1").fetchall()}
    src_orders, wh_orders = sum(v[0] for v in src.values()), sum(v[0] for v in wh.values())
    src_rev, wh_rev = sum(v[1] for v in src.values()), sum(v[1] for v in wh.values())
    add("reconcile_order_count", abs(src_orders - wh_orders),
        f"source={src_orders} warehouse={wh_orders}")
    add("reconcile_revenue", abs(src_rev - wh_rev),
        f"source_cents={src_rev} warehouse_cents={wh_rev}")
    bad_dates = sorted(d for d in set(src) | set(wh) if src.get(d) != wh.get(d))
    add("reconcile_by_date", len(bad_dates), f"dates differ={bad_dates[:5]}")
    return r

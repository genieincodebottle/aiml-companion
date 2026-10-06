"""The DuckDB warehouse. Three schemas, one rule.

    build      scratch copy of the published tables, changed by one run
    published  what readers query, replaced only by `publish`
    ops        run history, check results, publish log

The rule is that nothing writes `published` except `publish`, which does it in
one transaction. A run that fails anywhere earlier leaves `published` exactly as
the last good run left it.
"""
import hashlib

import duckdb

PUBLISHED_TABLES = ("dim_customer_scd2", "fct_order", "fct_order_line")

OPS_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS published;
CREATE TABLE IF NOT EXISTS ops.runs (
    run_id VARCHAR, run_date DATE, seq INTEGER, task VARCHAR,
    status VARCHAR, attempts INTEGER, error VARCHAR);
CREATE TABLE IF NOT EXISTS ops.check_results (
    run_id VARCHAR, run_date DATE, stage VARCHAR, check_name VARCHAR,
    passed BOOLEAN, detail VARCHAR);
CREATE TABLE IF NOT EXISTS ops.publish_log (
    run_id VARCHAR, run_date DATE, fct_order_rows BIGINT, revenue_cents BIGINT);
"""


class Warehouse:
    def __init__(self, path=":memory:"):
        self.conn = duckdb.connect(str(path))
        self.conn.execute(OPS_DDL)

    def exists(self, schema, table):
        return self.conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?", [schema, table]).fetchone()[0] == 1

    # ---- run history -------------------------------------------------------
    def next_run_id(self, run_date):
        n = self.conn.execute(
            "SELECT count(DISTINCT run_id) FROM ops.runs WHERE run_date = ?",
            [run_date]).fetchone()[0]
        return f"{run_date.isoformat()}#{n + 1}"

    def record_task(self, run_id, run_date, seq, result):
        self.conn.execute("INSERT INTO ops.runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                          [run_id, run_date, seq, result.task, result.status,
                           result.attempts, result.error])

    def record_checks(self, run_id, run_date, results):
        for r in results:
            self.conn.execute("INSERT INTO ops.check_results VALUES (?, ?, ?, ?, ?, ?)",
                              [run_id, run_date, r.stage, r.name, r.passed, r.detail])

    # ---- build and publish -------------------------------------------------
    def reset_build(self):
        """Start from a copy of the last published state, so a run can fail safely."""
        self.conn.execute("DROP SCHEMA IF EXISTS build CASCADE")
        self.conn.execute("CREATE SCHEMA build")
        for table in PUBLISHED_TABLES:
            if self.exists("published", table):
                self.conn.execute(
                    f"CREATE TABLE build.{table} AS SELECT * FROM published.{table}")

    def publish(self, run_id, run_date):
        """Swap every published table to the build version in one transaction."""
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for table in PUBLISHED_TABLES:
                self.conn.execute(f"DROP TABLE IF EXISTS published.{table}")
                self.conn.execute(
                    f"CREATE TABLE published.{table} AS SELECT * FROM build.{table}")
            rows, revenue = self.conn.execute(
                "SELECT count(*), coalesce(sum(revenue_cents), 0) FROM published.fct_order"
            ).fetchone()
            self.conn.execute("INSERT INTO ops.publish_log VALUES (?, ?, ?, ?)",
                              [run_id, run_date, rows, revenue])
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ---- reads used by checks, reports and tests ---------------------------
    def checksum(self, schema, table):
        """sha256 of the table's rows in a total order. Equal checksum, equal table."""
        rows = self.conn.execute(f"SELECT * FROM {schema}.{table} ORDER BY ALL").fetchall()
        return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()

    def checksums(self, schema="published"):
        return {t: self.checksum(schema, t) for t in PUBLISHED_TABLES
                if self.exists(schema, t)}

    def revenue_by_date_region(self, schema="published"):
        """[(order_date text, region, revenue_cents)] for revenue-counting orders."""
        if not self.exists(schema, "fct_order"):
            return []
        rows = self.conn.execute(
            f"SELECT CAST(order_date AS VARCHAR), region, sum(revenue_cents) "
            f"FROM {schema}.fct_order WHERE revenue_cents > 0 "
            f"GROUP BY ALL ORDER BY ALL").fetchall()
        return [(d, r, int(c)) for d, r, c in rows]

    def published_revenue(self):
        if not self.exists("published", "fct_order"):
            return 0
        return int(self.conn.execute(
            "SELECT coalesce(sum(revenue_cents), 0) FROM published.fct_order").fetchone()[0])

    def runs(self, run_id=None):
        where, args = ("WHERE run_id = ?", [run_id]) if run_id else ("", [])
        return self.conn.execute(
            f"SELECT run_id, seq, task, status, attempts, error FROM ops.runs {where} "
            "ORDER BY run_date, CAST(split_part(run_id, '#', 2) AS INTEGER), seq", args).fetchall()

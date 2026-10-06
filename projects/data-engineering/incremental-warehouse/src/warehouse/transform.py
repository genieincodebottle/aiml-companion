"""Run the SQL models in `sql/`. They are plain SELECT files with a little Jinja.

`{{ ref('x') }}` becomes `build.x`, the same idea as dbt. The two flags that
change the SQL (`detect_deletes`, `scd_type`) come from the pipeline options, so
a naive variant uses the same file as the correct one.
"""
from pathlib import Path

import jinja2

def read_model(name):
    sql_dir = Path(__file__).resolve().parents[2] / "sql"
    return (sql_dir / f"{name}.sql").read_text(encoding="utf-8")


def render(name, cfg, opts):
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    statuses = ", ".join(f"'{s}'" for s in cfg["revenue_statuses"])
    return env.from_string(read_model(name)).render(
        ref=lambda model: f"build.{model}",
        revenue_statuses=statuses,
        detect_deletes=opts.detect_deletes,
        scd_type=opts.scd_type)


def build_table(conn, name, cfg, opts):
    """Materialise a model as build.<name>. Returns its row count."""
    conn.execute(f"CREATE OR REPLACE TABLE build.{name} AS {render(name, cfg, opts)}")
    return conn.execute(f"SELECT count(*) FROM build.{name}").fetchone()[0]

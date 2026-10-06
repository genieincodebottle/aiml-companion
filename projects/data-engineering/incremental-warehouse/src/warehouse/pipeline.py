"""The DAG for one logical date, and the tasks in it.

    extract_orders, extract_customers
      >> land_raw >> check_raw >> prepare_build
      >> stg_orders, stg_order_lines, stg_customer_versions
      >> dim_customer_scd2 >> plan_partitions
      >> fct_order, fct_order_line >> check_marts >> publish

Airflow mapping. Each `add(...)` is a PythonOperator, `run_date` is
`{{ ds }}`, and `>>` is the same operator. `check_raw` and `check_marts` are
tasks that raise, so downstream tasks are skipped, which is Airflow's
`all_success` rule. `publish` is the last task, so a failed check means nothing
is published.
"""
from dataclasses import dataclass

from warehouse import checks, extract, load, transform
from warehouse.checks import CheckFailed
from warehouse.dag import DAG
from warehouse.land import TABLES


@dataclass
class Env:
    """Everything a task can touch. The pipeline never receives the world or its answer."""
    cfg: dict
    opts: object
    shop: object
    store: object
    wh: object
    clock: object


def _partition_date(ctx):
    """The date a run files its raw data under. The wall-clock option is the bug."""
    env = ctx.env
    return ctx.run_date if env.opts.clock == "run_date" else env.clock.today()


def task_extract_orders(ctx):
    return extract.extract_orders(ctx.env.shop.conn, ctx.run_date, ctx.env.opts)


def task_extract_customers(ctx):
    return extract.extract_customers(ctx.env.shop.conn, ctx.run_date, ctx.env.opts)


def task_land_raw(ctx):
    part = _partition_date(ctx)
    landed = {**ctx.xcom["extract_orders"], **ctx.xcom["extract_customers"]}
    for table, arrow in landed.items():
        ctx.env.store.land(table, part, arrow)
    return {"partition": part, "rows": {t: a.num_rows for t, a in landed.items()}}


def task_check_raw(ctx):
    env, part = ctx.env, ctx.xcom["land_raw"]["partition"]
    results = checks.check_raw(env.store, env.cfg, ctx.run_date, part)
    return _settle(ctx, results, quarantine=part)


def _settle(ctx, results, quarantine=None):
    """Record results. In gate mode a failure stops the run."""
    env = ctx.env
    env.wh.record_checks(ctx.run_id, ctx.run_date, results)
    failed = [r for r in results if not r.passed]
    if failed and env.opts.checks_mode == "gate":
        if quarantine is not None:
            env.store.quarantine(quarantine, TABLES)
        raise CheckFailed("; ".join(f"{r.name} ({r.detail})" for r in failed))
    return {"failed": [r.name for r in failed]}


def task_prepare_build(ctx):
    ctx.env.wh.reset_build()
    ctx.env.store.register(ctx.env.wh.conn)


def _model(name):
    def task(ctx):
        env = ctx.env
        return transform.build_table(env.wh.conn, name, env.cfg, env.opts)
    return task


def task_plan_partitions(ctx):
    env = ctx.env
    touched = load.plan_touched_dates(env.wh.conn, ctx.xcom["land_raw"]["partition"],
                                      env.opts.detect_deletes, env.opts.scd_type)
    return [d.isoformat() for d in touched]


def _fact(name):
    def task(ctx):
        env = ctx.env
        sql = transform.render(name, env.cfg, env.opts)
        return load.load_partitions(env.wh.conn, name, sql, env.opts.load_mode,
                                    ctx.xcom["land_raw"]["partition"])
    return task


def task_check_marts(ctx):
    env = ctx.env
    results = checks.check_marts(env.wh.conn, env.shop.conn, env.cfg)
    return _settle(ctx, results)


def task_publish(ctx):
    ctx.env.wh.publish(ctx.run_id, ctx.run_date)


def build_dag(opts):
    dag = DAG("incremental_warehouse", opts.retries, opts.retry_delay_seconds,
              opts.retry_backoff)
    t = {name: dag.add(name, fn) for name, fn in [
        ("extract_orders", task_extract_orders), ("extract_customers", task_extract_customers),
        ("land_raw", task_land_raw), ("check_raw", task_check_raw),
        ("prepare_build", task_prepare_build),
        ("stg_orders", _model("stg_orders")),
        ("stg_order_lines", _model("stg_order_lines")),
        ("stg_customer_versions", _model("stg_customer_versions")),
        ("dim_customer_scd2", _model("dim_customer_scd2")),
        ("plan_partitions", task_plan_partitions),
        ("fct_order", _fact("fct_order")), ("fct_order_line", _fact("fct_order_line")),
        ("check_marts", task_check_marts), ("publish", task_publish)]}
    [t["extract_orders"], t["extract_customers"]] >> t["land_raw"] >> t["check_raw"]
    t["check_raw"] >> t["prepare_build"]
    t["prepare_build"] >> [t["stg_orders"], t["stg_order_lines"], t["stg_customer_versions"]]
    t["stg_customer_versions"] >> t["dim_customer_scd2"]
    [t["stg_orders"], t["stg_order_lines"], t["dim_customer_scd2"]] >> t["plan_partitions"]
    t["plan_partitions"] >> [t["fct_order"], t["fct_order_line"]]
    [t["fct_order"], t["fct_order_line"]] >> t["check_marts"] >> t["publish"]
    return dag

"""The DAG runner on its own, with toy tasks."""
import pytest

from warehouse.dag import DAG, RunContext, TransientError


def _run(dag, sleeps=None):
    log = []
    ctx = RunContext("r1", "2025-01-01", None)
    results = dag.run(ctx, log.append, sleep=(sleeps.append if sleeps is not None else lambda s: None))
    return {r.task: r for r in results}, ctx, log


def test_rshift_wires_dependencies_and_order_is_repeatable():
    dag = DAG("t")
    a, b, c, d = (dag.add(n, lambda ctx, n=n: n) for n in "abcd")
    [a, b] >> c >> d
    assert [t.name for t in dag.order()] == ["a", "b", "c", "d"]
    assert {u.name for u in c.upstream} == {"a", "b"}


def test_cycle_is_rejected():
    dag = DAG("t")
    a, b = dag.add("a", lambda ctx: 1), dag.add("b", lambda ctx: 1)
    a >> b >> a
    with pytest.raises(ValueError, match="cycle"):
        dag.order()


def test_every_task_sees_the_logical_date_and_passes_values_downstream():
    dag = DAG("t")
    a = dag.add("a", lambda ctx: ctx.run_date)
    b = dag.add("b", lambda ctx: ctx.xcom["a"] + "!")
    a >> b
    results, ctx, _ = _run(dag)
    assert ctx.xcom["b"] == "2025-01-01!"
    assert all(r.status == "success" for r in results.values())


def test_transient_error_is_retried_with_doubling_delay():
    calls = []

    def flaky(ctx):
        calls.append(ctx.attempt)
        if ctx.attempt < 3:
            raise TransientError("reset")
        return "ok"

    dag = DAG("t", retries=2, retry_delay=0.05, backoff=2.0)
    dag.add("flaky", flaky)
    sleeps = []
    results, _, _ = _run(dag, sleeps)
    assert calls == [1, 2, 3]
    assert results["flaky"].status == "success" and results["flaky"].attempts == 3
    assert sleeps == [0.05, 0.1]


def test_retries_run_out_and_the_task_fails():
    dag = DAG("t", retries=1)

    def always(ctx):
        raise TransientError("down")

    dag.add("x", always)
    results, _, _ = _run(dag)
    assert results["x"].status == "failed" and results["x"].attempts == 2


def test_non_transient_error_is_not_retried_and_downstream_is_skipped():
    calls = []

    def bad(ctx):
        calls.append(ctx.attempt)
        raise RuntimeError("bad data")

    dag = DAG("t", retries=3)
    a, b, c = dag.add("a", bad), dag.add("b", lambda ctx: 1), dag.add("c", lambda ctx: 1)
    a >> b >> c
    results, _, log = _run(dag)
    assert calls == [1]
    assert [results[n].status for n in "abc"] == ["failed", "skipped", "skipped"]
    assert results["a"].error == "RuntimeError: bad data"
    assert len(log) == 3

"""GATE 3: percentiles, goodput, the knee, cost and fleet size. Fails until analysis/ is written."""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.cost import cost_per_million_tokens
from analysis.fleet import concurrent_requests, fleet_plan, replicas_needed
from analysis.goodput import goodput, max_slo_concurrency, meets_slo, slo_attainment
from analysis.knee import find_knee, plot_knee
from analysis.tables import latency_table, percentile, throughput
from bench.records import RequestRecord


def rec(rid, start, ttft, e2e, tpot, itl=None, out=100, ok=True):
    return RequestRecord(rid=rid, start_s=start, ttft_s=ttft, e2e_s=e2e, tpot_s=tpot, itl_s=itl or [],
                         prompt_tokens=50, output_tokens=out, ok=ok)


# ---- percentiles ---------------------------------------------------------------------
def test_percentile_matches_numpy_and_does_not_sort_in_place():
    vals = [9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0]
    keep = list(vals)
    for p in (0, 25, 50, 90, 99, 100):
        assert percentile(vals, p) == pytest.approx(float(np.percentile(vals, p)))
    assert vals == keep
    with pytest.raises(ValueError):
        percentile([], 50)


def test_itl_is_pooled_across_requests_not_averaged_per_request():
    # 99 smooth requests and one request that stalls once for 2 seconds in a 100-gap answer.
    smooth = [rec(i, i, 0.1, 5, 0.05, itl=[0.05] * 100) for i in range(99)]
    stall = rec(99, 99, 0.1, 7, 0.07, itl=[0.05] * 99 + [2.0])
    t = latency_table(smooth + [stall])
    assert t["n_ok"] == 100 and t["n_failed"] == 0
    assert t["itl_s"][99] == pytest.approx(0.05)
    assert t["tpot_s"][50] == pytest.approx(0.05)
    assert set(t["ttft_s"]) == {50, 90, 95, 99}
    # 10,000 pooled gaps, one of them 2.0 s: only the very top of the distribution sees it
    t2 = latency_table(smooth + [stall], percentiles=(100,))
    assert t2["itl_s"][100] == pytest.approx(2.0), "p100 of pooled ITL must see the stall"
    per_request_mean_max = max(sum(r.itl_s) / len(r.itl_s) for r in smooth + [stall])
    assert per_request_mean_max < 0.1, "(this is why averaging per request hides the stall)"


def test_failed_requests_are_counted_and_excluded():
    recs = [rec(0, 0, 0.2, 2, 0.02), rec(1, 1, 0.4, 3, 0.03), rec(2, 2, 0, 0, 0, ok=False)]
    t = latency_table(recs)
    assert t["n_ok"] == 2 and t["n_failed"] == 1
    assert t["ttft_s"][50] == pytest.approx(0.3)
    empty = latency_table([rec(0, 0, 0, 0, 0, ok=False)])
    assert empty["n_ok"] == 0 and empty["ttft_s"] == {}


def test_throughput_uses_the_observed_span():
    recs = [rec(0, 0.0, 0.1, 10.0, 0.1, out=100), rec(1, 5.0, 0.1, 10.0, 0.1, out=300), rec(2, 6.0, 0, 0, 0, ok=False)]
    th = throughput(recs)           # span = 0 .. 15
    assert th["req_per_s"] == pytest.approx(2 / 15)
    assert th["output_tok_per_s"] == pytest.approx(400 / 15)
    assert throughput(recs, duration_s=20.0)["output_tok_per_s"] == pytest.approx(20.0)
    assert throughput([])["req_per_s"] == 0


# ---- goodput -------------------------------------------------------------------------
def test_goodput_counts_only_requests_that_meet_the_slo():
    good = [rec(i, i, 0.5, 5, 0.05) for i in range(6)]
    slow_first = [rec(10 + i, i, 3.0, 8, 0.05) for i in range(2)]
    slow_decode = [rec(20 + i, i, 0.5, 12, 0.4) for i in range(2)]
    allr = good + slow_first + slow_decode
    assert meets_slo(good[0], 2.0, 0.15) and not meets_slo(slow_first[0], 2.0, 0.15)
    assert not meets_slo(slow_decode[0], 2.0, 0.15)
    assert meets_slo(rec(0, 0, 2.0, 5, 0.15), 2.0, 0.15), "limits are inclusive"
    assert slo_attainment(allr, 2.0, 0.15) == pytest.approx(0.6)
    assert slo_attainment(allr + [rec(99, 0, 0, 0, 0, ok=False)], 2.0, 0.15) == pytest.approx(6 / 11)
    gp = goodput(allr, 2.0, 0.15, duration_s=10.0)
    assert gp == pytest.approx(0.6)
    assert gp < throughput(allr, duration_s=10.0)["req_per_s"]
    assert slo_attainment([], 2.0, 0.15) == 0.0


def test_max_slo_concurrency_picks_the_highest_level_that_still_qualifies():
    def level(c, frac_good):
        n = 40
        good = int(n * frac_good)
        return {"concurrency": c, "records": [rec(i, i, 0.5 if i < good else 9.0, 5, 0.05) for i in range(n)]}
    levels = [level(1, 1.0), level(4, 1.0), level(16, 0.97), level(32, 0.6), level(64, 0.1)]
    assert max_slo_concurrency(levels, 2.0, 0.15) == 16
    assert max_slo_concurrency(levels, 2.0, 0.15, min_attainment=0.99) == 4
    assert max_slo_concurrency([level(8, 0.1)], 2.0, 0.15) is None


# ---- knee ----------------------------------------------------------------------------
def test_find_knee():
    assert find_knee([(1, 100), (2, 190), (4, 340), (8, 400), (16, 410)]) == 8
    assert find_knee([(1, 100), (2, 200), (4, 400)]) == 4
    assert find_knee([(1, 100), (2, 105)]) == 1
    assert find_knee([(3, 50)]) == 3
    with pytest.raises(ValueError):
        find_knee([])


def test_plot_knee_writes_a_png(tmp_path):
    pts = [{"concurrency": c, "output_tok_per_s": t, "p99_ttft_s": l, "p99_itl_s": l / 10}
           for c, t, l in [(1, 100, 0.2), (4, 340, 0.5), (16, 410, 3.0), (64, 415, 20.0)]]
    out = tmp_path / "knee.png"
    plot_knee(pts, str(out), knee=4, title="test")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 4000


# ---- cost and fleet ------------------------------------------------------------------
def test_cost_per_million_tokens():
    assert cost_per_million_tokens(1000, 2.00) == pytest.approx(2.0 / 3600 / 1000 * 1e6)
    assert cost_per_million_tokens(1000, 2.00, num_gpus=4) == pytest.approx(4 * 0.5555555556, rel=1e-6)
    assert cost_per_million_tokens(1000, 2.00, utilisation=0.5) == pytest.approx(2 * cost_per_million_tokens(1000, 2.0))
    assert cost_per_million_tokens(500, 0.0) == 0.0
    for bad in (dict(tokens_per_s=0, gpu_hourly_usd=2), dict(tokens_per_s=10, gpu_hourly_usd=-1),
                dict(tokens_per_s=10, gpu_hourly_usd=2, utilisation=0), dict(tokens_per_s=10, gpu_hourly_usd=2, utilisation=1.5)):
        with pytest.raises(ValueError):
            cost_per_million_tokens(**bad)


def test_littles_law_and_replicas():
    assert concurrent_requests(10, 4) == 40
    assert replicas_needed(10, 4, 16, headroom=0.25) == 4
    assert replicas_needed(10, 4, 16, headroom=0.0) == 3
    assert replicas_needed(0.01, 1, 16) == 1
    with pytest.raises(ValueError):
        replicas_needed(10, 4, 0)
    with pytest.raises(ValueError):
        replicas_needed(10, 4, 16, headroom=1.0)
    with pytest.raises(ValueError):
        concurrent_requests(-1, 4)
    plan = fleet_plan(10, 4, 16, gpus_per_replica=2, headroom=0.25)
    assert plan["replicas"] == 4 and plan["gpus"] == 8 and plan["in_flight"] == 40
    assert plan["utilisation_at_peak"] == pytest.approx(40 / 64)
    assert not math.isnan(plan["utilisation_at_peak"])

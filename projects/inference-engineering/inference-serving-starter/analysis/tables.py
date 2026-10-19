"""Percentile tables and throughput. YOU implement all three functions.

A mean hides the thing you are paid to find. Report percentiles, and be careful which
population each one is computed over.
"""
from __future__ import annotations

from bench.records import RequestRecord

PERCENTILES = (50, 90, 95, 99)


def percentile(values: list, p: float) -> float:
    """YOU. The p-th percentile (0..100) with linear interpolation between order statistics,
    the same convention as numpy.percentile's default. Raise ValueError on an empty list.
    Do not sort the caller's list in place."""
    raise NotImplementedError("analysis/tables.py: percentile")


def latency_table(records: list, percentiles=PERCENTILES) -> dict:
    """YOU. Summarise a list of RequestRecord.

    Return:
        {"n_ok": int, "n_failed": int,
         "ttft_s": {50: .., 90: .., ...}, "tpot_s": {...}, "e2e_s": {...}, "itl_s": {...}}

    Rules (tests/test_analysis.py checks each one):
      * Failed requests (ok False) are counted in n_failed and excluded from every percentile.
      * ttft_s, tpot_s and e2e_s are per-request: one number per OK request.
      * itl_s is POOLED: every inter-token gap of every OK request goes into one population,
        so a request with 400 tokens contributes 399 samples. This is why p99 ITL is not the
        p99 of per-request mean ITL. A stall inside one long answer must show up.
      * Keys of each inner dict are the integer percentiles passed in.
      * No OK requests: return n_ok 0 and empty inner dicts, do not raise.
    """
    raise NotImplementedError("analysis/tables.py: latency_table")


def throughput(records: list, duration_s: float | None = None) -> dict:
    """YOU. {"req_per_s": .., "output_tok_per_s": .., "input_tok_per_s": ..} over OK requests.

    duration_s None means the observed span: from the earliest start_s to the latest
    (start_s + e2e_s) among the OK requests. Output tokens use record.output_tokens.
    Return zeros when there are no OK requests or the span is not positive.
    """
    raise NotImplementedError("analysis/tables.py: throughput")

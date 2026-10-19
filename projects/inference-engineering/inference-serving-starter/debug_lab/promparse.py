"""Tiny Prometheus text-format helpers for the incident bundles. GIVEN.

Enough to read the scrapes in debug_lab/incidents/ and a live /metrics endpoint. It is not
a full parser: no escaped quotes in label values, no timestamps, no exemplars.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional

_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)\s*$')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


class Sample(NamedTuple):
    name: str
    labels: dict
    value: float


def parse_prom(text: str) -> list:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, lab, val = m.groups()
        labels = dict(_LABEL.findall(lab)) if lab else {}
        out.append(Sample(name, labels, float(val)))
    return out


def metric_value(text: str, name: str, **labels) -> Optional[float]:
    """Sum of all samples called `name` whose labels include `labels`. None if absent."""
    vals = [s.value for s in parse_prom(text)
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())]
    return sum(vals) if vals else None


def hist_mean(text: str, base: str) -> Optional[float]:
    s, c = metric_value(text, base + "_sum"), metric_value(text, base + "_count")
    return None if not c else s / c


def hist_quantile(text: str, base: str, q: float) -> Optional[float]:
    """PromQL-style histogram_quantile over the cumulative buckets of one scrape."""
    buckets = []
    for s in parse_prom(text):
        if s.name == base + "_bucket":
            le = s.labels["le"]
            buckets.append((float("inf") if le == "+Inf" else float(le), s.value))
    buckets.sort()
    if not buckets or buckets[-1][1] == 0:
        return None
    rank = q * buckets[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0.0)
        prev_le, prev_c = le, c
    return None


def load_scrapes(scrape_dir) -> list:
    """[(t_seconds, text), ...] sorted by time, from files named scrape_NN_tNNNNs.prom."""
    out = []
    for p in sorted(Path(scrape_dir).glob("scrape_*.prom")):
        m = re.search(r"_t(\d+)s", p.name)
        out.append((int(m.group(1)) if m else len(out), p.read_text(encoding="utf-8")))
    return out

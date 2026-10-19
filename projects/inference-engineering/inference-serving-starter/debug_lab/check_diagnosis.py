"""The GIVEN diagnosis checker. It does not grade your prose.

It checks two things only:
  1. ROOT_CAUSE names the mechanism (keyword groups below, all must match).
  2. EVIDENCE cites real numbers: each line must name a metric that exists in the scrape
     file you point at, and the value you quote must match that file.

Your diagnosis file (debug_lab/diagnoses/<incident>.md) has this shape:

    ROOT_CAUSE: one sentence naming the mechanism
    EVIDENCE:
    - vllm:kv_cache_usage_perc | scrapes/scrape_08_t0270s.prom | 0.98
    - vllm:num_preemptions_total | scrapes/scrape_10_t0330s.prom | 412
    - vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total | scrapes/scrape_12_t0390s.prom | 0.5
    FIX: what you would change, and what you would measure to confirm it

A metric field may be a ratio "a / b". Paths are relative to the incident folder, for
example replica1/scrapes/scrape_08_t0270s.prom. Counters are cumulative, so say which scrape
you read. Values must match within 2 percent (or 0.01 absolute).

    python debug_lab/check_diagnosis.py            # check every incident
    python debug_lab/check_diagnosis.py a_kv_saturation
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from promparse import metric_value  # noqa: E402

V = "vllm:"
SPEC = {
    "a_kv_saturation": {
        "causes": [[r"kv"], [r"preempt", r"saturat", r"exhaust", r"\bfull\b", r"capacity", r"out of"]],
        "metrics": {V + "kv_cache_usage_perc", V + "num_preemptions_total", V + "num_requests_waiting"},
        "min_distinct": 2, "min_dirs": 1,
    },
    "b_hot_prefix_imbalance": {
        "causes": [[r"imbalanc", r"skew", r"uneven", r"\bhot", r"concentrat"],
                   [r"prefix", r"affinity", r"rout", r"hash"]],
        "metrics": {V + "num_requests_waiting", V + "num_requests_running", V + "kv_cache_usage_perc",
                    V + "num_preemptions_total", V + "prefix_cache_hits_total", V + "generation_tokens_total"},
        "min_distinct": 2, "min_dirs": 2,   # evidence must come from BOTH replicas
    },
    "c_speculation_collapse": {
        "causes": [[r"specul", r"n-?gram", r"draft"],
                   [r"compute", r"batch", r"concurren", r"ridge", r"verif"]],
        "metrics": {V + "num_requests_running", V + "num_requests_waiting",
                    V + "spec_decode_num_accepted_tokens_total", V + "spec_decode_num_draft_tokens_total",
                    V + "spec_decode_num_drafts_total", V + "generation_tokens_total"},
        "min_distinct": 2, "min_dirs": 1, "needs_spec_metric": True,
    },
}


class Result(NamedTuple):
    ok: bool
    reasons: list


def _eval_metric(expr: str, text: str):
    parts = [p.strip() for p in expr.split("/")]
    vals = [metric_value(text, p) for p in parts]
    if any(v is None for v in vals):
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2 and vals[1]:
        return vals[0] / vals[1]
    return None


def check(incident: str, diag_dir: Path | None = None, inc_dir: Path | None = None) -> Result:
    spec = SPEC[incident]
    diag_dir = diag_dir or ROOT / "diagnoses"
    inc_dir = (inc_dir or ROOT / "incidents") / incident
    path = diag_dir / f"{incident}.md"
    if not path.exists():
        return Result(False, [f"missing {path.relative_to(ROOT.parent) if path.is_relative_to(ROOT.parent) else path}"])
    text = path.read_text(encoding="utf-8")
    reasons: list = []

    m = re.search(r"^ROOT_CAUSE:\s*(.+)$", text, re.M)
    if not m:
        reasons.append("no 'ROOT_CAUSE: ...' line")
    else:
        rc = m.group(1).lower()
        for group in spec["causes"]:
            if not any(re.search(p, rc) for p in group):
                reasons.append(f"ROOT_CAUSE does not name the mechanism (need one of: {', '.join(group)})")

    ev_lines = []
    in_ev = False
    for line in text.splitlines():
        if line.startswith("EVIDENCE:"):
            in_ev = True
            continue
        if re.match(r"^[A-Z_]+:", line):
            in_ev = False
        if in_ev and line.strip().startswith("-"):
            ev_lines.append(line.strip()[1:].strip())
    good_metrics: set = set()
    dirs: set = set()
    spec_metric_seen = False
    if not ev_lines:
        reasons.append("no EVIDENCE lines ('- metric | scrape file | value')")
    for ln in ev_lines:
        cols = [c.strip() for c in ln.split("|")]
        if len(cols) != 3:
            reasons.append(f"bad evidence line (need 3 columns): {ln[:60]}")
            continue
        metric, rel, claimed = cols
        f = inc_dir / rel
        if not f.exists():
            reasons.append(f"scrape file not found: {rel}")
            continue
        actual = _eval_metric(metric, f.read_text(encoding="utf-8"))
        if actual is None:
            reasons.append(f"metric not in {rel}: {metric}")
            continue
        try:
            c = float(claimed)
        except ValueError:
            reasons.append(f"value is not a number: {claimed}")
            continue
        if abs(actual - c) > max(0.01, 0.02 * abs(actual)):
            reasons.append(f"{metric} in {rel} is {actual:.4g}, you wrote {claimed}")
            continue
        names = {p.strip() for p in metric.split("/")}
        if names & spec["metrics"]:
            good_metrics |= names & spec["metrics"]
            dirs.add(rel.split("/")[0] if "/" in rel and rel.split("/")[0].startswith("replica") else ".")
            if any("spec_decode" in n for n in names):
                spec_metric_seen = True
    if len(good_metrics) < spec["min_distinct"]:
        reasons.append(f"cite at least {spec['min_distinct']} distinct discriminating metrics with correct values "
                       f"(got {len(good_metrics)})")
    if len(dirs) < spec["min_dirs"]:
        reasons.append("evidence must come from both replicas" if spec["min_dirs"] == 2 else "no evidence directory")
    if spec.get("needs_spec_metric") and not spec_metric_seen:
        reasons.append("cite at least one vllm:spec_decode_* metric (or the accepted/draft ratio)")
    if not re.search(r"^FIX:\s*\S+", text, re.M):
        reasons.append("no 'FIX: ...' line")
    return Result(not reasons, reasons)


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(SPEC)
    rc = 0
    for n in names:
        r = check(n)
        print(("PASS " if r.ok else "FAIL ") + n)
        for why in r.reasons:
            print("   - " + why)
        rc |= 0 if r.ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

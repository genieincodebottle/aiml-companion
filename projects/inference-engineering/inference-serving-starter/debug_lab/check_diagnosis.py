"""The GIVEN diagnosis checker. It does not grade your prose.

It checks two things only:
  1. ROOT_CAUSE names the mechanism (keyword groups below, all must match).
  2. EVIDENCE cites real numbers: each line must name a metric that exists in the scrape
     file you point at, and the value you quote must match that file.
  3. FIX says what to change AND what would confirm it: a metric name and a direction.

Counters only mean something as a difference, so a counter you cite must come from at
least two different scrapes of the same folder. A diagnosis also needs two or more
distinct scrape files in total, and at least one counter.

Your diagnosis file (debug_lab/diagnoses/<incident>.md) has this shape:

    ROOT_CAUSE: one sentence naming the mechanism
    EVIDENCE:
    - vllm:kv_cache_usage_perc | scrapes/scrape_08_t0270s.prom | 0.98
    - vllm:num_preemptions_total | scrapes/scrape_10_t0330s.prom | 412
    - vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total | scrapes/scrape_12_t0390s.prom | 0.5
    FIX: what you would change. Confirm by checking that vllm:num_preemptions_total stops rising and
         vllm:kv_cache_usage_perc falls below 0.9 after the change.

A metric field may be a ratio "a / b". Paths are relative to the incident folder, for
example replica1/scrapes/scrape_08_t0270s.prom. Counters are cumulative, so cite the same
counter from two scrapes. Values must match within 2 percent (or 0.01 absolute). The FIX block
is the FIX: line plus any lines under it until the next KEY: line. A metric name may be written
with or without the vllm: prefix there.

    python debug_lab/check_diagnosis.py            # check every incident
    python debug_lab/check_diagnosis.py a_kv_saturation

d_config_regression is OPTIONAL (a before/after regression with a config diff). The checker
skips it when you have not written debug_lab/diagnoses/d_config_regression.md.
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
        "min_distinct": 2, "min_dirs": 1, "min_scrapes": 2,
    },
    "b_hot_prefix_imbalance": {
        "causes": [[r"imbalanc", r"skew", r"uneven", r"\bhot", r"concentrat"],
                   [r"prefix", r"affinity", r"rout", r"hash"]],
        "metrics": {V + "num_requests_waiting", V + "num_requests_running", V + "kv_cache_usage_perc",
                    V + "num_preemptions_total", V + "prefix_cache_hits_total", V + "generation_tokens_total"},
        "min_distinct": 2, "min_dirs": 2, "min_scrapes": 2,   # evidence must come from BOTH replicas
    },
    "c_speculation_collapse": {
        "causes": [[r"specul", r"n-?gram", r"draft"],
                   [r"compute", r"batch", r"concurren", r"ridge", r"verif"]],
        "metrics": {V + "num_requests_running", V + "num_requests_waiting",
                    V + "spec_decode_num_accepted_tokens_total", V + "spec_decode_num_draft_tokens_total",
                    V + "spec_decode_num_drafts_total", V + "generation_tokens_total"},
        "min_distinct": 2, "min_dirs": 1, "min_scrapes": 2, "needs_spec_metric": True,
    },
    "d_config_regression": {
        # before/ and after/ hold the same traffic against two configs; the serve_args diff says what changed.
        "causes": [[r"max[-_ ]?num[-_ ]?seqs", r"seq(uence)?s? (cap|limit)", r"concurrency (cap|limit)",
                    r"running (cap|limit)", r"batch (cap|limit)"],
                   [r"queue", r"waiting", r"regress", r"cap", r"limit", r"config", r"diff"]],
        "metrics": {V + "num_requests_running", V + "num_requests_waiting", V + "kv_cache_usage_perc",
                    V + "num_preemptions_total", V + "generation_tokens_total"},
        "min_distinct": 3, "min_dirs": 2, "min_scrapes": 2,   # evidence from BOTH before/ and after/
    },
}
OPTIONAL = {"d_config_regression"}
COUNTER_SUFFIX = "_total"
CONFIRM_WORD = re.compile(r"\b(confirm|verif|validat)\w*", re.I)
DIRECTION_WORD = re.compile(r"\b(fall|falls|drop|drops|decreas\w*|rise|rises|increas\w*|recover\w*|return\w*|"
                            r"stay|stays|below|above|under|over|flat|zero|stop|stops|plateau\w*)\b|[<>]", re.I)


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
    scrapes: set = set()                 # distinct (folder, file) pairs with a verified value
    counter_files: dict = {}             # (counter name, folder) -> set of scrape files cited
    counter_seen = False
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
        parts = rel.split("/")
        folder = "/".join(parts[:-2]) if len(parts) > 2 and parts[-2] == "scrapes" else "."
        for n in names:
            if n.endswith(COUNTER_SUFFIX):
                counter_files.setdefault((n, folder), set()).add(parts[-1])
        if names & spec["metrics"]:
            good_metrics |= names & spec["metrics"]
            dirs.add(folder)
            scrapes.add((folder, parts[-1]))
            counter_seen = counter_seen or any(n.endswith(COUNTER_SUFFIX) for n in names & spec["metrics"])
            if any("spec_decode" in n for n in names):
                spec_metric_seen = True
    if len(good_metrics) < spec["min_distinct"]:
        reasons.append(f"cite at least {spec['min_distinct']} distinct discriminating metrics with correct values "
                       f"(got {len(good_metrics)})")
    if len(dirs) < spec["min_dirs"]:
        reasons.append("evidence must come from both folders (replicas, or before/ and after/)"
                       if spec["min_dirs"] == 2 else "no evidence directory")
    if len(scrapes) < spec["min_scrapes"]:
        reasons.append(f"cite at least {spec['min_scrapes']} different scrape files (got {len(scrapes)}): "
                       "one snapshot cannot show a trend")
    if not counter_seen:
        reasons.append("cite at least one counter (a name ending in _total), because a counter is the rate of change")
    for name in sorted({n for n, _ in counter_files}):
        if not any(len(f) >= 2 for (n, _), f in counter_files.items() if n == name):
            reasons.append(f"{name} is a counter: cite it from two different scrapes of the same folder "
                           "so a difference (a rate) exists")
    if spec.get("needs_spec_metric") and not spec_metric_seen:
        reasons.append("cite at least one vllm:spec_decode_* metric (or the accepted/draft ratio)")
    fix = re.search(r"^FIX:\s*(\S.*(?:\n(?![A-Z_]+:).*)*)", text, re.M)
    if not fix:
        reasons.append("no 'FIX: ...' line")
    else:
        block = fix.group(1)
        short = {n[len(V):] if n.startswith(V) else n for n in spec["metrics"]}
        names_in_fix = [s for s in short if re.search(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", block)]
        if not CONFIRM_WORD.search(block):
            reasons.append("FIX does not say what would confirm it (use the word 'confirm' or 'verify')")
        if not names_in_fix:
            reasons.append("FIX does not name the metric you would watch to confirm it "
                           f"(for example {sorted(spec['metrics'])[0]})")
        if not DIRECTION_WORD.search(block):
            reasons.append("FIX does not say which way that metric should move (falls, rises, stays below ...)")
    return Result(not reasons, reasons)


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(SPEC)
    rc = 0
    for n in names:
        if n in OPTIONAL and not (ROOT / "diagnoses" / f"{n}.md").exists():
            print(f"SKIP {n} (optional, no diagnosis file)")
            continue
        r = check(n)
        print(("PASS " if r.ok else "FAIL ") + n)
        for why in r.reasons:
            print("   - " + why)
        rc |= 0 if r.ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

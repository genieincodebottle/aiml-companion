"""OPTIONAL extension: a quality-equivalence gate for a changed serving configuration. GIVEN.

You changed the config to go faster: fp8 or int4 weights, n-gram speculation, a bigger batch.
Throughput says it is faster. It does not say the answers are still the same. This compares the
outputs of a CANDIDATE config with a BASELINE config on one fixed prompt set (quality/prompts.jsonl),
both produced at temperature 0 by quality/run_prompts.py, and reports how often they agree.

    python quality/check_equivalence.py --baseline quality/outputs/bf16.jsonl \
        --candidate quality/outputs/ngram.jsonl --min-exact 0.9 --min-prefix 0.95

Metrics (all on whitespace-separated words, so no tokenizer is needed):
  exact     fraction of prompts whose output is word-for-word identical
  prefix    mean over prompts of (length of the common leading words / length of the longer output)
  diverge   median word index of the first difference, over the prompts that differ

What the thresholds mean. They are YOUR decision, so there is no default pass mark: you must pass
--min-exact and --min-prefix, and the report says why you chose them.
  * Speculative decoding should not change greedy output. The vLLM docs say speculative decoding is
    lossless up to hardware numerics and that greedy sampling with it matches greedy sampling without
    it (https://docs.vllm.ai/en/latest/features/speculative_decoding/, read 2026-11-16). They also
    say batch-size changes can vary logprobs, so a few late divergences are expected on a real GPU:
    a high bar (for example --min-exact 0.9) is a defensible choice there.
  * fp8, int4 or AWQ weights DO change the outputs. A pass mark here is a statement about how much
    change you accept before you run a real evaluation on your own task. Agreement on 32 prompts is
    a smoke test, not an accuracy measurement: it cannot tell you the quality is the same.

Exit code 0 when both thresholds are met, 1 when not, 2 for a malformed input.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load(path) -> dict:
    """{id: {"kind": str, "text": str}}. One JSON object per line with id, kind and text."""
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "text"):
            if key not in row:
                raise ValueError(f"{path}:{n}: missing {key!r}")
        out[row["id"]] = {"kind": row.get("kind", ""), "text": row["text"]}
    return out


def common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(baseline: dict, candidate: dict) -> dict:
    """Agreement of candidate against baseline on the prompts both have. Raises ValueError if the
    two files cover different prompts (a comparison on different prompts is meaningless)."""
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must contain the same prompt ids "
                         f"(only in baseline: {sorted(set(baseline) - set(candidate))[:3]}, "
                         f"only in candidate: {sorted(set(candidate) - set(baseline))[:3]})")
    if not baseline:
        raise ValueError("no prompts to compare")
    rows, firsts = [], []
    for pid in sorted(baseline):
        a, b = baseline[pid]["text"].split(), candidate[pid]["text"].split()
        cp = common_prefix(a, b)
        longest = max(len(a), len(b))
        same = a == b
        rows.append({"id": pid, "kind": baseline[pid]["kind"], "exact": same,
                     "prefix": 1.0 if longest == 0 else cp / longest})
        if not same:
            firsts.append(cp)
    by_kind: dict = {}
    for r in rows:
        k = by_kind.setdefault(r["kind"] or "-", {"n": 0, "exact": 0, "prefix": 0.0})
        k["n"] += 1
        k["exact"] += r["exact"]
        k["prefix"] += r["prefix"]
    return {
        "n": len(rows),
        "exact": sum(r["exact"] for r in rows) / len(rows),
        "prefix": statistics.fmean(r["prefix"] for r in rows),
        "diverge_median_word": statistics.median(firsts) if firsts else None,
        "by_kind": {k: {"n": v["n"], "exact": v["exact"] / v["n"], "prefix": v["prefix"] / v["n"]}
                    for k, v in sorted(by_kind.items())},
        "worst": [r["id"] for r in sorted(rows, key=lambda r: r["prefix"])[:3] if not r["exact"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--min-exact", type=float, required=True, help="minimum fraction of identical outputs")
    ap.add_argument("--min-prefix", type=float, required=True, help="minimum mean common-prefix fraction")
    a = ap.parse_args(argv)
    try:
        res = compare(load(a.baseline), load(a.candidate))
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{res['n']} prompts: exact {res['exact']:.3f}, prefix {res['prefix']:.3f}, "
          f"median first-difference word {res['diverge_median_word']}")
    for k, v in res["by_kind"].items():
        print(f"  {k:<7} n={v['n']:<3} exact {v['exact']:.2f}  prefix {v['prefix']:.2f}")
    if res["worst"]:
        print("  least similar:", ", ".join(res["worst"]))
    ok = res["exact"] >= a.min_exact and res["prefix"] >= a.min_prefix
    print("PASS" if ok else f"FAIL (needs exact >= {a.min_exact}, prefix >= {a.min_prefix})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

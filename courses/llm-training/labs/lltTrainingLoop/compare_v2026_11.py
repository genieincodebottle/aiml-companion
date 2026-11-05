"""Compare the re-measured T1 runs (runs/lltTrainingLoop/_v2026-11, torch 2.15.0, numpy 2.5.4) with the
earlier ones (torch 2.10.0, numpy 2.4.2), then file them under the version policy.

Unchanged values (every loss within 5e-5, the last digit the page prints): the new file replaces the
old one under the same runId, so the env it records is the one that produced the numbers.
Changed values: the new file is added as <runId>-v2026-11 and the old one stays.
Usage: python compare_v2026_11.py [--apply]
"""
import json
import os
import shutil
import sys

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltTrainingLoop"))
NEW = os.path.join(RUNS, "_v2026-11")
TOL = 5e-5


def numbers(x, path=""):
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return
    if isinstance(x, (int, float)):
        yield path, float(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, path + "." + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, path + "[" + str(i) + "]")


SKIP = ("elapsedSec", "wallClockSec", "tokPerSec", "trainSec")
apply = "--apply" in sys.argv
for f in sorted(os.listdir(NEW)):
    if not f.endswith(".json"):
        continue
    new = json.load(open(os.path.join(NEW, f), encoding="utf-8"))
    old_path = os.path.join(RUNS, f)
    if not os.path.exists(old_path):
        print("NEW  ", f, "(no earlier run)")
        if apply:
            shutil.copy(os.path.join(NEW, f), old_path)
        continue
    old = json.load(open(old_path, encoding="utf-8"))
    a = {p: v for p, v in numbers({"metrics": old.get("metrics"), "evals": old.get("evals")}) if not any(s in p for s in SKIP)}
    b = {p: v for p, v in numbers({"metrics": new.get("metrics"), "evals": new.get("evals")}) if not any(s in p for s in SKIP)}
    common = set(a) & set(b)
    worst = max(((abs(a[p] - b[p]), p) for p in common), default=(0.0, ""))
    text_old = {k: v for k, v in (old.get("evals") or {}).items() if isinstance(v, str)}
    text_new = {k: v for k, v in (new.get("evals") or {}).items() if isinstance(v, str)}
    changed = worst[0] > TOL or set(a) != set(b) or text_old != text_new
    if text_old != text_new:
        print("  printed output differs in", sorted(k for k in text_new if text_new.get(k) != text_old.get(k)))
    print("CHANGED" if changed else "same   ", f, "| values", len(common), "| max diff %.3g at %s" % worst,
          "| key sets differ" if set(a) != set(b) else "")
    if apply:
        if changed:
            new["runId"] = new["runId"] + "-v2026-11"
            json.dump(new, open(os.path.join(RUNS, new["runId"] + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        else:
            shutil.copy(os.path.join(NEW, f), old_path)

"""Data for the near-duplicate threshold widget (llt5/lltInstructionData.TRYIT_SPEC.md).

From the notebook run's prompt embeddings: for every threshold 0.80 to 0.99 in steps of 0.01, the rows the greedy
pass drops (granite, and MiniLM for the English-only switch), by source and language; and candidate example pairs
per band for a hand review. Usage: python make_tryit_data.py <notebook data dir> <out.json> [--candidates]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
data, out = Path(sys.argv[1]), Path(sys.argv[2])
pool = [json.loads(line) for line in open(data / "pool_scored.jsonl", encoding="utf-8")]
F = json.loads((data / "flags.json").read_text(encoding="utf-8"))
E = np.load(data / "prompt_embeddings.npy")
exact = set(F["exact_dup"])
keep = [i for i in range(len(pool)) if i not in exact]


def group(r):
    if r["source"] != "wildchat":
        return r["source"]
    return "wildchat-english" if r["lang"] == "English" else "wildchat-other"


def sweep(Emat, idx):
    S = Emat[idx] @ Emat[idx].T
    rows = []
    for t100 in range(80, 100):
        t = t100 / 100
        kept, dropped = [], []
        for j in range(len(idx)):
            if kept and S[j, kept].max() >= t:
                dropped.append(idx[j])
            else:
                kept.append(j)
        rows.append({"threshold": t, "dropped": len(dropped), "kept": len(idx) - len(dropped),
                     "droppedByGroup": dict(collections.Counter(group(pool[i]) for i in dropped))})
        print(t, len(dropped), flush=True)
    return rows


S = E[keep] @ E[keep].T
np.fill_diagonal(S, -1.0)
nn, arg = S.max(axis=1), S.argmax(axis=1)
hist = np.histogram(nn, bins=np.arange(0.70, 1.0001, 0.01))
res = {"source": "lltInstructionData-T2-dedup (notebook run llt-insdata-notebook v1)", "rows": len(keep),
       "exactDuplicatesRemovedFirst": len(exact), "groups": dict(collections.Counter(group(pool[i]) for i in keep)),
       "nearestNeighbourHistogram": {"binStart": [round(float(b), 2) for b in hist[1][:-1]], "count": hist[0].tolist()},
       "granite": sweep(E, keep)}

if "--candidates" in sys.argv:
    pii = set(map(int, F["pii"])) | set(map(int, F["names"]))
    cand = collections.defaultdict(list)
    for j in np.argsort(-nn):
        a, b = keep[j], keep[arg[j]]
        if a in pii or b in pii or a > b:
            continue
        pa, pb = pool[a]["prompt"].strip(), pool[b]["prompt"].strip()
        if len(pa) > 160 or len(pb) > 160:
            continue
        band = round(float(np.floor(nn[j] * 100) / 100), 2)
        if band >= 0.85 and len(cand[band]) < 6:
            cand[band].append({"sim": round(float(nn[j]), 3), "a": pa, "b": pb, "aGroup": group(pool[a]), "bGroup": group(pool[b])})
    for band in sorted(cand):
        print("==", band)
        for c in cand[band]:
            print("  ", c["sim"], c["aGroup"], repr(c["a"][:110]), "||", c["bGroup"], repr(c["b"][:110]))
    res["candidates"] = cand

out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out)

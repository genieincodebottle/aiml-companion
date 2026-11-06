"""Broken run: an English-only embedder used for near-dup detection on a multilingual pool.

Compares all-MiniLM-L6-v2 (English, local CPU) with the notebook's granite-embedding-311m-multilingual-r2
(embeddings saved by the Kaggle run) on the same 7,990 Dolci and WildChat rows. Same greedy rule, same order.
Usage: python measure_embedders.py <dir with the notebook's data/>   (encodes the local pool with MiniLM on the CPU once, cached in data/)
"""
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np

from runlog import write_run

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
data = Path(sys.argv[1])
pool = [json.loads(line) for line in open(data / "pool_scored.jsonl", encoding="utf-8")]
n = 7990
assert all(not r["id"].startswith("synth") for r in pool[:n]) and pool[n]["id"].startswith("synth")
local = [json.loads(line) for line in open(HERE / "data" / "pool.jsonl", encoding="utf-8")]
assert [r["id"] for r in local] == [r["id"] for r in pool[:n]], "local pool differs from the Kaggle pool"
cache = HERE / "data" / "E_all-MiniLM-L6-v2.npy"
if not cache.exists():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    np.save(cache, m.encode([r["prompt"] for r in local], batch_size=32, normalize_embeddings=True, convert_to_numpy=True))
E_mini = np.load(cache)
E_gran = np.load(data / "prompt_embeddings.npy")[:n]


def lang(r):
    return r.get("lang") or ("dolci" if r["source"] == "dolci" else "synthetic")


def norm(s):
    return " ".join(s.lower().split())


first, exact = {}, set()
for i, r in enumerate(pool[:n]):
    k = norm(r["prompt"])
    if k in first:
        exact.add(i)
    else:
        first[k] = i
keep = [i for i in range(n) if i not in exact]


def greedy(E, t):
    S = E[keep] @ E[keep].T
    kept, dropped = [], []
    for j in range(len(keep)):
        if kept and S[j, kept].max() >= t:
            dropped.append(keep[j])
        else:
            kept.append(j)
    return dropped


metrics, samples = [], []
for name, E in [("all-MiniLM-L6-v2", E_mini), ("granite-embedding-311m-multilingual-r2", E_gran)]:
    for t in [0.85, 0.9, 0.95]:
        d = greedy(E, t)
        langs = collections.Counter(lang(pool[i]) for i in d)
        metrics.append({"embedder": name, "threshold": t, "dropped": len(d), "droppedByLanguage": dict(langs.most_common()),
                        "nonEnglishWildChatShare": round(sum(v for k, v in langs.items() if k not in ("English", "dolci", "synthetic")) / max(1, len(d)), 3)})
        print(metrics[-1])
zh = [i for i, r in enumerate(pool[:n]) if r.get("lang") == "Chinese"]


def example_pair():
    a = next(i for i in zh if pool[i]["prompt"].strip() == "你是谁")
    b = next(i for i in zh if pool[i]["prompt"].strip() == "来首诗")
    return {"a": "你是谁 (who are you)", "b": "来首诗 (write a poem)",
            "miniLM": round(float(E_mini[a] @ E_mini[b]), 3), "granite": round(float(E_gran[a] @ E_gran[b]), 3)}


Sz = E_mini[zh] @ E_mini[zh].T
np.fill_diagonal(Sz, -1)
jz, sz = Sz.argmax(1), Sz.max(1)
hit = np.flatnonzero(sz >= 0.95)
g = np.array([float(E_gran[zh[i]] @ E_gran[zh[jz[i]]]) for i in hit])
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
unk = float(np.mean([np.mean(np.array(tk(pool[k]["prompt"][:300])["input_ids"]) == tk.unk_token_id) for k in zh[:300]]))
metrics.append({"chinesePrompts": len(zh), "chineseWithMiniLMNeighbourAtLeast095": int(len(hit)),
                "graniteMedianOnThosePairs": round(float(np.median(g)), 3), "graniteShareAtLeast095OnThosePairs": round(float((g >= 0.95).mean()), 3),
                "miniLMUnknownTokenShareInFirst300ChinesePrompts": round(unk, 3),
                "examplePair": example_pair()})
print(metrics[-1])
write_run("lltInstructionData-T1-broken-english-embedder", "T1", "broken:english-only-embedder",
          {"rows": n, "exactDuplicatesRemovedFirst": len(exact), "rule": "greedy in pool order, drop a row if cosine >= threshold to a kept row",
           "miniLM": "sentence-transformers/all-MiniLM-L6-v2, max_seq_length 256, local CPU",
           "granite": "embeddings saved by the notebook run (T4), max_seq_length 512"}, 0, metrics)

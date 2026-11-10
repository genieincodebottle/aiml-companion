"""Paired intervals the lltShip page quotes that the notebook does not print, computed from the per-item scores in the
Run Library files. Writes runs/lltShip/lltShip-T3-analysis.json.

    python analyze_runs.py

Bits per byte needs each text's UTF-8 byte count, which the run files do not store. The script rebuilds the same 200
Hindi and 200 English texts with the panel recipe of code cell 4 (deterministic streaming order, same cut, same
1,023-token filter) and checks that the rebuilt counts reproduce every recorded bpb before it uses them.
Same bootstrap as cell 4's ci(): 10,000 resamples of the items, seed 0, middle 95% of (b minus a).
"""
import json, math, sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run, RUNS_DIR  # noqa: E402

load = lambda name: json.loads((Path(RUNS_DIR) / (name + ".json")).read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")


def cut(text, n_chars):
    return text[:n_chars].rsplit(" ", 1)[0] if len(text) > n_chars else None


def held_texts(name, config, split, n_docs, n_chars, want):
    out = []
    for r in load_dataset(name, name=config, split=split, streaming=True).take(n_docs):
        t = cut(r["text"], n_chars)
        if t and len(tok(t).input_ids) <= 1023:
            out.append(t)
        if len(out) == want:
            break
    return out


H_BYTES = np.array([len(t.encode("utf-8")) for t in held_texts("HuggingFaceFW/fineweb-2", "hin_Deva", "test", 400, 1000, 200)])
E_BYTES = np.array([len(t.encode("utf-8")) for t in held_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 500, 3000, 200)])

M = {}
for f in ("lltShip-T3-panel-parents", "lltShip-T3-merge-grid", "lltShip-T3-gguf", "lltShip-T3-gptq", "lltShip-T3-qat"):
    d = load(f)
    for k, v in d["evals"].items():
        if k in d["evals"].get("items", {}):
            M[k] = dict(v, items={kk: np.array(vv, dtype=float) for kk, vv in d["evals"]["items"][k].items()})
grid = load("lltShip-T3-merge-grid")
best = grid["config"]["best"]

# Self-check: the rebuilt byte counts must give back every recorded bpb
worst = max(max(abs(m["items"]["hi"].sum() / math.log(2) / H_BYTES.sum() - m["hiBpb"]),
                abs(m["items"]["en"].sum() / math.log(2) / E_BYTES.sum() - m["enBpb"])) for m in M.values())
assert worst < 1e-9, worst


def ci(a, b, key, n=10000, seed=0):
    num = {"hi": ("hi", H_BYTES), "en": ("en", E_BYTES), "reply": ("reply", "replyTok"), "ended": ("ended", None)}[key]
    rng = np.random.default_rng(seed)
    N = len(a["items"][num[0]])
    i = rng.integers(0, N, (n, N))

    def stat(m):
        top = m["items"][num[0]][i].sum(1)
        if num[1] is None:
            return top / N
        den = num[1][i].sum(1) * math.log(2) if isinstance(num[1], np.ndarray) else m["items"][num[1]][i].sum(1)
        return top / den
    d = stat(b) - stat(a)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


merges = [k for k in grid["evals"] if k != "items"]
out = {
    "replyVsBase": {k: ci(M["base"], M[k], "reply") for k in merges},
    "hiVsHindiModel": {k: ci(M["hindi"], M[k], "hi") for k in merges},
    "hiVsBase": {k: ci(M["base"], M[k], "hi") for k in merges},
    "enVsBase": {k: ci(M["base"], M[k], "en") for k in merges + ["sft", "hindi"]},
    "endedVsBase": {k: ci(M["base"], M[k], "ended") for k in merges + ["sft", "hindi"]},
    "shipped": {"name": best, "hiVsHindi": ci(M["hindi"], M[best], "hi"), "replyVsSft": ci(M["sft"], M[best], "reply"),
                "hiVsBase": ci(M["base"], M[best], "hi"), "replyVsBase": ci(M["base"], M[best], "reply"),
                "enVsBase": ci(M["base"], M[best], "en"), "endedVsSft": ci(M["sft"], M[best], "ended")},
    "runnerUp": {"name": "task arithmetic 1.0 / 0.25", "hiMinusShipped": ci(M[best], M["task arithmetic 1.0 / 0.25"], "hi"),
                 "replyMinusShipped": ci(M[best], M["task arithmetic 1.0 / 0.25"], "reply")},
    "dareSeeds": {"hi": ci(M["DARE p 0.9"], M["DARE p 0.9 seed 1"], "hi"), "reply": ci(M["DARE p 0.9"], M["DARE p 0.9 seed 1"], "reply")},
    "gptqOwnMinusGgufQ4KM": {"hi": ci(M["gguf Q4_K_M"], M["gptq own"], "hi"), "reply": ci(M["gguf Q4_K_M"], M["gptq own"], "reply")},
    "qatVsMerged": {"hi": ci(M[best], M["qat 3-bit"], "hi"), "reply": ci(M[best], M["qat 3-bit"], "reply"),
                    "en": ci(M[best], M["qat 3-bit"], "en")},
    "qatMinusQ3KM": {"hi": ci(M["gguf Q3_K_M"], M["qat 3-bit"], "hi"), "reply": ci(M["gguf Q3_K_M"], M["qat 3-bit"], "reply")},
    "compressedVsBase": {k: {"hi": ci(M["base"], M[k], "hi"), "reply": ci(M["base"], M[k], "reply")}
                         for k in ("gguf Q8_0", "gguf Q4_K_M", "gguf Q3_K_M", "gptq own", "gptq wikitext", "qat 3-bit")},
    "bytesCheckMaxGap": worst,
}
write_run("lltShip-T3-analysis", "T3", "analysis", {"bootstrap": 10000, "seed": 0, "from": ["lltShip-T3-panel-parents",
          "lltShip-T3-merge-grid", "lltShip-T3-gguf", "lltShip-T3-gptq", "lltShip-T3-qat"]}, 0, [], evals=out,
          notes="Computed on a laptop from the recorded per-item scores; no model was run.")
print(json.dumps(out, indent=1))

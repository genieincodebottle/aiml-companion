"""Build tryit_data.json for the optional lltShip Try it widget (see TRYIT_SPEC.md) from the Run Library files.

    python tryit_data.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "lltShip"
load = lambda name: json.loads((RUNS / (name + ".json")).read_text(encoding="utf-8"))
parents, grid = load("lltShip-T3-panel-parents"), load("lltShip-T3-merge-grid")
gguf, gptq, qat = load("lltShip-T3-gguf"), load("lltShip-T3-gptq"), load("lltShip-T3-qat")
KEYS = ("hiBpb", "enBpb", "replyNll", "endRate", "meanTokens")
pick = lambda m: {k: round(m[k], 4) for k in KEYS if k in m}


def method(name):
    for prefix, m in (("average", "task arithmetic"), ("task arithmetic", "task arithmetic"), ("TIES", "TIES"), ("DARE", "DARE")):
        if name.startswith(prefix):
            return m


out = {
    "source": ["lltShip-T3-panel-parents", "lltShip-T3-merge-grid", "lltShip-T3-gguf", "lltShip-T3-gptq", "lltShip-T3-qat"],
    "panel": grid["config"],
    "parents": {k: pick(v) for k, v in parents["evals"].items() if k != "items"},
    "merges": [dict(pick(v), name=k, method=method(k), kept=v["kept"]) for k, v in grid["evals"].items() if k != "items"],
    "shipped": grid["config"]["best"],
    "compressed": [dict(pick(v), name=k, MiB=v.get("MiB")) for d in (gguf, gptq, qat) for k, v in d["evals"].items()
                   if isinstance(v, dict) and "hiBpb" in v],
}
(HERE / "tryit_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("merges", len(out["merges"]), "compressed", len(out["compressed"]))

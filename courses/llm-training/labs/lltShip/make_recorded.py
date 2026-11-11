"""Build recorded.json, the small file the notebook downloads when FROM_OUR_CHECKPOINT = True (steps 5, 7, 9 and 12 print
these numbers instead of recomputing them). Every value comes from our Run Library files in runs/lltShip/ (A100 runs).

    python make_recorded.py            writes recorded.json next to this file
    python upload_recorded.py          pushes it to aimlcompanion/llm-training-lab under lltShip/
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs" / "lltShip"
load = lambda n: json.load(open(RUNS / ("lltShip-T3-" + n + ".json"), encoding="utf-8"))["evals"]


def panel(m):
    keep = ("hiBpb", "enBpb", "replyNll", "endRate", "meanTokens", "kept", "MiB", "sec", "kinds")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in keep}


grid = load("merge-grid")
gguf = load("gguf")
gptq = load("gptq")
qat = load("qat")
vllm = load("vllm")["vllm"]
rec = {
    "about": "Recorded on a Google Cloud A100 40 GB VM with the same notebook cells. Run files: lltShip-T3-merge-grid, -gguf, -gptq, -qat, -vllm.",
    "ship": "task arithmetic 1.0 / 0.5",
    "grid": {k: panel(v) for k, v in grid.items() if k != "items"},
    "gguf": {"panels": {k[5:]: panel(v) for k, v in gguf.items() if k.startswith("gguf ")},
             "files": {q: {kk: vv for kk, vv in f.items() if kk != "file"} for q, f in gguf["files"].items()}},
    "gptq": {k[5:]: panel(v) for k, v in gptq.items() if k.startswith("gptq ")},
    "qat": {"rtn": panel(qat["rtn 3-bit"]), "qat": panel(qat["qat 3-bit"]),
            "klFirst": qat["qat"]["klFirst"], "klLast": qat["qat"]["klLast"], "trainSec": qat["qat"]["trainSec"],
            "peakGB": qat["qat"]["peakGB"], "steps": 150},
    "vllm": vllm,
}
out = Path(__file__).resolve().parent / "recorded.json"
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes", "| merges", len(rec["grid"]), "| gguf panels", sorted(rec["gguf"]["panels"]))

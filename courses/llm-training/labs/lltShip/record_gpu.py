"""Turn ship_<mode>.json files from run_gpu.py (Google Cloud VM llt-c3) into Run Library files in runs/lltShip/.

    python record_gpu.py <dir with ship_full.json, stdout_full.txt, ship_defaults.json>

One file per experiment group; per-item scores are kept so paired intervals and the Try it can be recomputed.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

src = Path(sys.argv[1])
full = json.loads((src / "ship_full.json").read_text(encoding="utf-8"))
R = full["runs"]
HW = {"gpu": full["gpu"], "count": 1, "vramGB": 40, "cpu": "Google Cloud a2-highgpu-1g, " + str(full["cpuCount"]) + " vCPU",
      "provider": "Google Cloud Spot VM llt-c3, us-central1-a"}
ENV = dict(full["versions"], cuda="13.0", llamaCpp="b12182", llmcompressor="0.15.0 (own venv, transformers 5.21.0)",
           vllm="0.33.0 (own venv, torch 2.14.0)")
NOTE = "Recorded on an A100 40 GB. A T4 runs the same code; T4 times on the page are estimates until re-timed."
PANEL = {"hiTexts": 200, "enTexts": 200, "replies": 300, "prompts": 100, "maxNewTokens": 256,
         "data": "FineWeb-2 hin_Deva test (lltDomainAdaptation panel), FineWeb-Edu sample-10BT, UltraFeedback test_prefs (lltDPO held-out pairs)"}


def slim(m):
    return {k: v for k, v in m.items() if k != "items"}


def rec(run_id, label, config, evals, keep_items=()):
    items = {k: R[k]["items"] for k in keep_items if k in R and "items" in R[k]}
    write_run(run_id, "T3", label, config, 0, [], evals=dict(evals, items=items) if items else evals,
              hardware=HW, env=ENV, notes=NOTE, wall_clock_sec=full["totalSec"])


parents = ["base", "sft", "hindi"]
grid = [k for k in R if k.startswith(("average", "task arithmetic", "TIES", "DARE"))]
rec("lltShip-T3-panel-parents", "baseline", PANEL, {k: slim(R[k]) for k in parents}, keep_items=parents)
rec("lltShip-T3-merge-grid", "ablation:merge_method", dict(PANEL, merges=grid, best=full["best"]),
    {k: slim(R[k]) for k in grid}, keep_items=grid)
gg = [k for k in R if k.startswith("gguf ")]
rec("lltShip-T3-gguf", "ablation:gguf_type", {"llamaCpp": "b12182", "types": gg}, dict({k: slim(R[k]) for k in gg}, files=R.get("gguf")),
    keep_items=gg)
gp = [k for k in R if k.startswith("gptq ")]
rec("lltShip-T3-gptq", "ablation:calibration", {"scheme": "W4A16", "rows": 128, "maxSeqLen": 512}, {k: slim(R[k]) for k in gp}, keep_items=gp)
qa = [k for k in R if k in ("rtn 3-bit", "qat 3-bit", "qat")]
rec("lltShip-T3-qat", "ablation:qat", {"bits": 3, "group": 32, "steps": 150, "seqPerStep": 8, "maxLen": 512, "lr": 1e-5,
                                       "loss": "forward KL to the unquantized merge"}, {k: slim(R[k]) for k in qa},
    keep_items=[k for k in qa if k != "qat"])
rec("lltShip-T3-serve", "baseline", {"server": "llama-server b12182, CPU", "file": "merged-Q4_K_M.gguf", "prompts": 20},
    {k: R[k] for k in R if k.startswith("serve ")})
if "vllm" in R:
    rec("lltShip-T3-vllm", "baseline", {"vllm": "0.33.0", "prompts": 100, "maxTokens": 256}, {"vllm": R["vllm"]})
for mode in ("full", "defaults"):
    p = src / ("ship_" + mode + ".json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        write_run("lltShip-T3-notebook-" + mode, "T3", "baseline", {"mode": mode, "cells": d["cells"]}, 0,
                  [{"step": 0, "peakGB": d["peakGB"]}], evals={"cellSec": d["cellSec"], "best": d["best"]},
                  hardware=HW, env=ENV, notes=NOTE, wall_clock_sec=d["totalSec"],
                  samples=[{"stdout": (src / ("stdout_" + mode + ".txt")).read_text(encoding="utf-8")}] if (src / ("stdout_" + mode + ".txt")).exists() else [])

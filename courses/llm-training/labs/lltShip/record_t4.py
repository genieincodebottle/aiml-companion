"""Record the 2026-11-11 T4 timing runs of lltShip as Run Library files (runs/lltShip/).

    python record_t4.py <dir>      <dir> holds off/ (run_gpu.py --mode defaults with the exit off: ship_defaults.json,
                                    stdout_defaults.txt, gpu.csv) and on/ (the real notebook run with nbconvert: out.ipynb, gpu.csv, DONE),
                                    and setup.json (install seconds, versions)

The VM was set up like Kaggle's GPU T4 x2: n1-highmem-4 (4 vCPU, 26 GB RAM) with two T4s, Python 3.12, torch 2.11.0+cu130.
"""
import csv, datetime, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

NLJ = chr(10)
src = Path(sys.argv[1])
setup = json.loads((src / "setup.json").read_text(encoding="utf-8"))
HW = {"gpu": "Tesla T4", "count": 2, "gpusUsed": 1, "vramGB": 15, "cpu": "Google Cloud n1-highmem-4, 4 vCPU, 26 GB RAM",
      "provider": "Google Cloud on-demand VM llt-c8, us-central1-b, set up like Kaggle's GPU T4 x2 (not Kaggle itself)"}
ENV = dict(setup["versions"], cuda="13.0", llamaCpp="b12182", llmcompressor="0.15.0 (own venv, transformers 5.21.0)",
           vllm="not run (RUN_VLLM = False)")
NAMES = {1: "cell01_step1.py", 2: "cell02_step2.py", 3: "cell03_step3.py", 4: "cell04_step4.py", 5: "cell05_step5.py", 6: "cell06_step6.py",
         7: "cell07_step7.py", 8: "cell08_step8.py", 9: "cell09_step9.py", 10: "cell10_step10.py", 11: "cell11_step11.py", 12: "cell12_step12.py"}


def smi_peak(path):
    rows = [r for r in csv.reader(open(path)) if len(r) == 3]
    return [round(max(int(r[i]) for r in rows) / 1024, 2) for i in (1, 2)]    # GiB per GPU, nvidia-smi memory.used


# Exit off: run_gpu.py --mode defaults, every cell in one process (same code as the notebook), torch's own peak
off = json.loads((src / "off" / "ship_defaults.json").read_text(encoding="utf-8"))
r = off["runs"]
write_run("lltShip-T2-gcpT4-notebook-nocheckpoint", "T2", "baseline",
          {"mode": "defaults, FROM_OUR_CHECKPOINT off (the old default path)", "cells": "1-12", "runGrid": False, "runQat": True, "runVllm": False},
          0, [{"step": 0, "peakGB": off["peakGB"]}],
          evals={"cellSec": off["cellSec"], "best": off["best"], "peakGBTorchAllocated": off["peakGB"],
                 "peakGiBNvidiaSmiPerGpu": smi_peak(src / "off" / "gpu.csv"),
                 "installSec": setup["installSec"], "panelSecEach": {k: r[k]["panelSec"] for k in ("average, lambda 0.5",)},
                 "qat": {k: r["qat"][k] for k in ("trainSec", "peakGB", "klFirst", "klLast")},
                 "gptqSec": {"wikitext": r["gptq wikitext"]["sec"], "own": r["gptq own"]["sec"]},
                 "ggufMakeSec": {q: g["sec"] for q, g in r["gguf"].items()}, "serve": {k: r[k] for k in ("serve ours", "serve chatml")}},
          hardware=HW, env=ENV, wall_clock_sec=off["totalSec"],
          notes="Install and first model downloads: install seconds are in evals.installSec, the downloads are inside cells 2 and 4 (cold cache). The old default: the equal average ships, QAT trains.",
          samples=[{"stdout": (src / "off" / "stdout_defaults.txt").read_text(encoding="utf-8")}])

# Exit on: the real notebook, headless
nb = json.loads((src / "on" / "out.ipynb").read_text(encoding="utf-8"))
p = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
steps = []
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code):
    ex = c["metadata"]["execution"]
    steps.append(round((p(ex["shell.execute_reply"]) - p(ex["iopub.execute_input"])).total_seconds(), 1))
# code cells: 0 = runtime check, 1 = install, 2.. = steps 1..12
on_wall = int((src / "on" / "DONE").read_text().split("wall")[1])
write_run("lltShip-T2-gcpT4-notebook", "T2", "baseline",
          {"mode": "the learner notebook with nbconvert, FROM_OUR_CHECKPOINT = True (the new default)", "cells": "runtime check, install, steps 1-12",
           "runGrid": False, "runQat": False, "runVllm": False},
          0, [{"step": 0, "peakGB": max(smi_peak(src / "on" / "gpu.csv"))}],
          evals={"cellSec": {"runtimeCheck": steps[0], "install": steps[1], **{NAMES[i]: s for i, s in enumerate(steps[2:], start=1)}},
                 "peakGiBNvidiaSmiPerGpu": smi_peak(src / "on" / "gpu.csv"), "note": "peakGB is nvidia-smi memory.used (includes the CUDA context and the caching allocator), not torch's allocated peak",
                 "notebookWallSec": on_wall},
          hardware=HW, env=ENV, wall_clock_sec=on_wall,
          samples=[{"stdout": NLJ.join("".join(o.get("text", "")) for c in code for o in c.get("outputs", []) if o["output_type"] == "stream")}],
          notes="Models were already in the Hugging Face cache from the exit-off run on the same VM, so step 2 and 4 downloads were warm. The install cell is a no-op here because setup had already installed the pins; setup.json has the cold install time.")

# The two optional switches, timed on their own (run_extras_t4.py for RUN_GRID, run_vllm_t4.py for RUN_VLLM), exit off
if (src / "x" / "extras_t4.json").exists():
    ex = json.loads((src / "x" / "extras_t4.json").read_text(encoding="utf-8"))
    vl = json.loads((src / "x" / "vllm_t4.json").read_text(encoding="utf-8"))
    write_run("lltShip-T2-gcpT4-grid-vllm", "T2", "ablation:optional_switches",
              {"runGrid": True, "runVllm": True, "fromOurCheckpoint": False, "note": "steps 1-5 with RUN_GRID, then steps 1-4 and step 12 with RUN_VLLM, in two processes"},
              0, [{"step": 0, "peakGB": max(ex["cellSec"]["5"]["peakGB"], vl["cellSec"]["cell04"]["peakGB"])}],
              evals={"grid": {"cellSec": ex["cellSec"], "totalSec": ex["totalSec"], "best": ex["best"], "merges": ex["grid"],
                              "peakGiBNvidiaSmiPerGpu": smi_peak(src / "x" / "gpu_grid.csv")},
                     "vllm": {"cellSec": vl["cellSec"], "totalSec": vl["totalSec"], "runs": vl["vllm"],
                              "peakGiBNvidiaSmiPerGpu": smi_peak(src / "x" / "gpu_vllm.csv"), "gpuMemoryUtilization": 0.5}},
              hardware=HW, env=dict(ENV, vllm="0.33.0 (own venv, torch 2.14.0)"), wall_clock_sec=round(ex["totalSec"] + vl["totalSec"], 1),
              notes="RUN_GRID and RUN_VLLM measured separately with the exit off. Step 12 includes building vllm-env, about 8 min of its 952 s.",
              samples=[{"stdout": (src / "x" / "run_grid.log").read_text(encoding="utf-8")[-6000:]},
                       {"stdout": (src / "x" / "run_vllm.log").read_text(encoding="utf-8")[-6000:]}])

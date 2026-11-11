"""Turn one T4 VM notebook run (out dir of run_notebook_timed.py: timing.json, result.json, stdout.txt, gpu.csv) into Run Library files.

  python record_gcp_t4.py <out dir> <run id> <label> [cpu <cpu-cells.json>]
Writes <run id>.json for each training run in result.json and <run id>-job.json (seconds per cell, install, evaluations, GPU peaks).
No new measurement. Environment of the 2026-11-11 T4 VM runs (llt-c7, us-central1-a).
"""
import csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

HW = {"gpu": "NVIDIA Tesla T4", "count": 1, "vramGB": 15.36,
      "host": "Google Cloud n1-highmem-4 (4 vCPU Intel Xeon 2.30 GHz, 26 GB RAM) with 2x T4 (the notebook uses one), on-demand, llt-c7; set up like Kaggle's GPU T4 x2"}
ENV = {"python": "3.12.16", "torch": "2.11.0+cu130", "cuda": "13.0", "transformers": "5.22.1", "trl": "1.17.1", "peft": "0.21.2", "datasets": "5.1.0",
       "accelerate": "1.15.0", "tokenizers": "0.23.3", "huggingface_hub": "2.4.0", "numpy": "2.5.4", "math_verify": "0.9.0",
       "nvidiaDriver": "580.178.04", "os": "Linux-6.8.0-1070-gcp-x86_64-with-glibc2.35 (Deep Learning VM image pytorch-2-9-cu129-ubuntu-2204-nvidia-580)",
       "source": "~/llt-venv (uv, Python 3.12.16, torch 2.11.0 from the cu130 index); the notebook's own %pip cell installed the Hugging Face pins"}

out, base_id, label = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if len(sys.argv) > 4 and sys.argv[4] == "cpu":
    c = json.load(open(sys.argv[5], encoding="utf-8"))
    write_run(base_id, "T1", "baseline", c["config"], 0, c["metrics"], samples=c["samples"], wall_clock_sec=c["wallClockSec"],
              hardware=dict(HW, gpu=None, count=0, vramGB=None, note="CUDA_VISIBLE_DEVICES empty: CPU only"), env=dict(ENV, cuda=None),
              notes="Steps 1, 2, 7 and 8 on the 4 vCPU T4 VM with the GPU hidden. measure_timing.py gave 0.19 min in three runs, models and data cached. First pass (run_steps.py) 11.4 s.")
    sys.exit(0)
T = json.load(open(out / "timing.json", encoding="utf-8"))
R = json.load(open(out / "result.json", encoding="utf-8")) if (out / "result.json").exists() else {}
txt = (out / "stdout.txt").read_text(encoding="utf-8", errors="replace")
peaks = {}
for r in csv.reader(open(out / "gpu.csv")):
    if r and not r[0].startswith("timestamp"):
        try:
            peaks[r[1].strip()] = max(peaks.get(r[1].strip(), 0), int(r[2].strip().split()[0]))
        except ValueError:
            pass
note = ("Measured on a Google Cloud VM set up like Kaggle's GPU T4 x2 (4 vCPU, 26 GB RAM, 2x T4, the notebook uses one). Notebook " + str(T["notebook"]) +
        " run headless with run_notebook_timed.py, substitutions " + json.dumps(T["subs"]) + ", default settings otherwise.")
written = []
for key, r in R.items():
    if not (isinstance(r, dict) and "hist" in r):
        continue
    rid = base_id + "-" + key
    ev = {k: v for k, v in r.items() if k not in ("cfg", "hist")}
    write_run(rid, "T2", label + ":" + key, r.get("cfg", {}), r.get("cfg", {}).get("seed", 0), r["hist"], evals=ev, wall_clock_sec=r.get("sec"),
              notes=note, hardware=HW, env=ENV)
    written.append(rid)
trl = re.search(r"(\d+) steps in ([0-9.]+) s [|] peak GB ([0-9.]+) [|] mean reward ([0-9.]+) [|] mean completion length ([0-9.]+)", txt)
job = dict(cells=T["cells"], totalCodeSec=T["totalCodeSec"], wallSec=T["wallSec"], error=T["error"], start=T["start"],
           nvidiaSmiPeakMiBPerGPU=peaks, base_eval=R.get("base_eval"), after=R.get("after"), mem=R.get("mem"))
if trl:
    job["trl"] = dict(steps=int(trl.group(1)), sec=float(trl.group(2)), peakGB=float(trl.group(3)), meanReward=float(trl.group(4)), meanCompletionLen=float(trl.group(5)))
write_run(base_id + "-job", "T2", label, {"runs": written, "subs": T["subs"]}, 0, [], evals=job, wall_clock_sec=T["wallSec"], notes=note, hardware=HW, env=ENV)

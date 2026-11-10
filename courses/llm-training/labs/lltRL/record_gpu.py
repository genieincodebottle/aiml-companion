"""Turn one Google Cloud job output (result.json + stdout.txt from gcp/llt-rl/run_job.py) into Run Library files. No new measurement.

  python record_gpu.py <job output dir> <run id>
A job with one run writes <run id>.json; a job with several writes lltRL-T2-<key>.json for each. The job as a whole
becomes <run id>-job.json
(cell times, session peak, the evaluations of the base model, the memory of the four PPO models).
The jobs ran on an A100 in fp16 autocast with GradScaler, the T4 recipe; wall-clock and memory are A100 numbers.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

out = Path(sys.argv[1])
base_id = sys.argv[2]
R = json.load(open(out / "result.json", encoding="utf-8"))
txt = (out / "stdout.txt").read_text(encoding="utf-8", errors="replace")
hw = {"gpu": R.get("gpu"), "count": 1, "vramGB": 40.0, "host": "Google Cloud a2-highgpu-1g Spot (12 vCPU, 85 GB RAM), llt-c3"}
env = dict(R["env"], accelerate="1.15.0", tokenizers="0.23.3", huggingface_hub="2.4.0", math_verify="0.9.0",
           source="~/llt-venv on the VM, recorded by run_job.py; HF pins from gcp/llt-rl/requirements.txt")
note = ("Recorded on an NVIDIA A100 with the T4 recipe (fp16 autocast, GradScaler, fp16 frozen bases). Timings and memory are A100 "
        "numbers; T4 times on the page are estimates until re-timed on a T4. Job " + out.name + ", mode " + R["mode"] + ", args " + json.dumps(R["args"]))


def strip(d, drop=("hist", "replies", "samples")):
    return {k: v for k, v in d.items() if k not in drop}


written = []
for key, r in R["runs"].items():
    run_id = base_id if len(R["runs"]) == 1 else "lltRL-T2-" + key
    cfg = dict(r.get("cfg", {}), job=out.name)
    label = "baseline" if key in ("ppo-kl0.05-s0", "ppo-kl0.05-s1", "grpo-main") and "dry" not in run_id else "ablation:" + key
    ev = {k: v for k, v in r.items() if k not in ("cfg", "hist")}
    write_run(run_id, "T2", label, cfg, cfg.get("seed", 0), r.get("hist", r.get("log", [])), evals=ev, wall_clock_sec=r.get("sec"),
              notes=note, hardware=hw, env=env)
    written.append(run_id)
job = {k: R.get(k) for k in ("mem", "base_eval", "after", "start", "end", "cellTimes", "sessionPeakGB", "totalSec") if k in R}
peaks = {m.group(1): float(m.group(2)) for m in re.finditer(r"CELL (cell[0-9]+_step[0-9]+[.]py) end [0-9.]+ [|] peak GB so far ([0-9.]+)", txt)}
job["peakGBAfterCell"] = peaks
write_run(base_id + "-job", "T2", "baseline", {"mode": R["mode"], "args": R["args"], "runs": written}, 0, [],
          evals=job, wall_clock_sec=R.get("totalSec"), notes=note, hardware=hw, env=env)
print("written", written)

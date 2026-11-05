"""Turn the Kaggle T4 output (kaggle/out/llt-attention-t4/llt_attention_t4.json) into Run Library files.

The kernel is kaggle/llt-attention-t4 (cells in kaggle/cells_attention_t4.py), pushed private with
--accelerator NvidiaTeslaT4. Timings from that machine count as measured: it is not shared.
"""
import json
from pathlib import Path

from runlog import write_run

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "kaggle" / "out" / "llt-attention-t4" / "llt_attention_t4.json"
R = json.loads(SRC.read_text(encoding="utf-8"))
hw = {"gpu": R["gpu"], "count": 1, "countOnMachine": R["gpu_count"], "vramGB": R["vram_gib"],
      "capability": R["capability"], "host": "Kaggle notebook, --accelerator NvidiaTeslaT4"}
env = {"python": R["python"], "torch": R["torch"], "cuda": R["cuda_runtime"], "os": "Kaggle image"}
wall = None
write_run("lltAttention-t2-t4-sweep", "T2", "experiment:naive-vs-sdpa",
          dict(R["config"], kernel="genieincodebottle/llt-attention-t4 (private), version 1",
               started=R["started"], finished=R["finished"]),
          seed=None, evals={"sweep": R["sweep"], "fp16Accuracy": R["fp16_accuracy"],
                            "fp16LeakageSdpaWorst": R["fp16_leakage_sdpa_worst"],
                            "sdpBackendsEnabled": {"flash": R["flash_sdp_enabled"], "memEfficient": R["mem_efficient_sdp_enabled"],
                                                   "math": R["math_sdp_enabled"]},
                            "flashWarning": "Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm 7.5 gpu."},
          hardware=hw, env=env, wall_clock_sec=wall)
write_run("lltAttention-t2-broken-fp16-mask", "T2", "broken:minus-1e9-in-fp16",
          {"kernel": "genieincodebottle/llt-attention-t4 (private), version 1", "T": 8, "D": 64,
           "padding": "position 0 is padding, so query 0 sees only masked keys"},
          evals=R["mask_value_fp16"], hardware=hw, env=env)

# The page's shipped T4 cell (cells/step_t4.py), run as kernel genieincodebottle/llt-attention-t4-cell.
LOG = HERE.parent.parent / "kaggle" / "out" / "llt-attention-t4-cell" / "llt-attention-t4-cell.log"
log = json.loads(LOG.read_text(encoding="utf-8"))
stdout = "".join(e["data"] for e in log if e["stream_name"] == "stdout")
rows, mode = [], None
for line in stdout.splitlines():
    if line.startswith("forward"):
        mode = line.strip()
    parts = line.split()
    if len(parts) == 5 and parts[0].isdigit():
        T, nm, sm, nt, st = int(parts[0]), *map(float, parts[1:])
        rows.append({"mode": mode, "T": T, "naiveMiB": nm, "sdpaMiB": sm, "naiveMs": nt, "sdpaMs": st})
assert len(rows) == 12, rows
write_run("lltAttention-t2-t4-cell", "T2", "experiment:naive-vs-sdpa",
          {"cell": "labs/lltAttention/cells/step_t4.py as shipped", "kernel": "genieincodebottle/llt-attention-t4-cell (private), version 1",
           "B": 4, "h": 12, "d": 64, "dtype": "float16", "iters": 20, "warmup": 3},
          evals={"table": rows}, hardware=hw, env=env, wall_clock_sec=round(log[-1]["time"] - log[0]["time"], 1), stdout=stdout)

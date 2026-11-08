"""Record the notebook kernel run (llt-rm-notebook v2) as Run Library file lltRewardModels-T2-notebook. No new measurement."""
import json, os, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402
OUT = HERE.parent.parent / "kaggle" / "out" / "llt-rm-notebook"
txt = (OUT / "notebook_stdout.txt").read_text(encoding="utf-8")
log = json.load(open(OUT / "llt-rm-notebook.log", encoding="utf-8"))
cells = {m.group(1): float(m.group(2)) for m in re.finditer(r"CELL (cell\d+_step\d) end ([\d.]+)", txt)}
steps = [{"step": int(a), "trainLoss75": float(b), "batchAcc75": float(c), "sec": int(d)}
         for a, b, c, d in re.findall(r"step (\d+) of 375 \| loss ([\d.]+) \| batch accuracy ([\d.]+) \| seconds (\d+)", txt)]
env = {"python": "3.13.16", "torch": "2.11.0+cu130", "transformers": "5.22.1", "trl": "1.17.1", "accelerate": "1.15.0",
       "datasets": "5.1.0", "tokenizers": "0.23.3", "huggingface_hub": "2.4.0", "source": "notebook install cell pins; runtime line of the kernel log"}
write_run("lltRewardModels-T2-notebook", "T2", "baseline",
          {"kernel": "llt-rm-notebook v2", "notebook": "labs/lltRewardModels/lltRewardModels.ipynb (same cells, plus a stdout tee)",
           "RUN_TRAINING": True, "RUN_EXPERIMENT": False}, 0, steps,
          evals={"cellEndSecAfterInstall": cells, "kernelSec": round(log[-1]["time"], 1), "stdout": txt},
          wall_clock_sec=round(log[-1]["time"], 1), hardware={"gpu": "Tesla T4", "count": 1, "vramGB": 15.0}, env=env,
          notes="Kaggle kernel llt-rm-notebook v2, 2026-11-07. v1 failed in its first second (the stdout tee lacked isatty).")

"""Record the step code blocks' run as Run Library files.

Usage: python from_steps_log.py <workdir> <steps stdout log>
Writes lltTrainingLoop-t1-cosine-s0 (the step 6 run, from <workdir>/small-s0.jsonl) and
lltTrainingLoop-t1-steps (the printed output of every code block, which the page quotes).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import write_run  # noqa: E402

work, log = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(os.path.join(work, "small-s0.jsonl"))]
cfg = dict(n_layer=4, n_head=4, d=128, block=64, dropout=0.0, micro_batch=12, accum=1, steps=2000, lr=1e-3, warmup=100,
           schedule="cosine", wd=0.1, clip=1.0, seed=0, eval_every=250, ckpt="small-s0.pt", ckpt_every=2000)
vals = [r["valLoss"] for r in rows if "valLoss" in r]
thin = [r for r in rows if r["step"] % 10 == 0 or "valLoss" in r or r["step"] <= 20]
write_run("lltTrainingLoop-t1-cosine-s0", "T1", "baseline", cfg, 0, thin,
          evals={"finalValLoss": vals[-1], "bestValLoss": min(vals), "finalTrainLoss": rows[-1]["trainLoss"],
                 "maxGradNorm": max(r["gradNorm"] for r in rows), "clippedSteps": sum(1 for r in rows if r["gradNorm"] > 1.0)},
          wall_clock_sec=rows[-1].get("elapsedSec"), notes="The step 6 code block run (steps/s6_run.py), deterministic CPU mode, 4 threads.")

blocks, cur = {}, None
for line in open(log, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("===== "):
        cur = line[6:]
        blocks[cur] = []
    elif line.startswith("----- done"):
        cur = None
    elif cur:
        blocks[cur].append(line)
write_run("lltTrainingLoop-t1-steps", "T1", "page-code-output", {"blocks": list(blocks)}, 0, [],
          evals={k: "\n".join(v) for k, v in blocks.items()},
          notes="Printed output of each page code block run in order by run_steps.py on CPU. Timings in it are distorted by other jobs.")

"""One GPU queue of the LIMA-style experiment (lltInstructionData, plan 7.8.3).

Usage: python worker_insdata.py <queue.json> <data_dir> <out_dir>
queue.json: [{"arm": "curated-1k", "seed": 0}, ..., {"arm": "ref-instruct"}]
<data_dir> holds pool_scored.jsonl and arms.json from the notebook run (steps 1 to 7).
Each job trains SmolLM2-360M on one arm with the page's own train_arm (cells/c08) and scores IFEval with the
page's ifeval_scores (cells/c09). Writes <out_dir>/<arm>-s<seed>.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
queue = json.load(open(sys.argv[1]))
data_dir, out_dir = Path(sys.argv[2]), Path(sys.argv[3])
wait_for = os.environ.get("LLT_WAIT_FOR")      # start only when the notebook has written its selection
while wait_for and not Path(wait_for).exists():
    time.sleep(30)
out_dir.mkdir(parents=True, exist_ok=True)

ns = {"__name__": "__main__"}
for f in ["c00_common.py", "c08_train.py", "c09_eval.py", "c10_functions.py"]:
    exec(compile((HERE / "worker_cells" / f).read_text(encoding="utf-8"), f, "exec"), ns)
torch, np = ns["torch"], ns["np"]
pool = [json.loads(line) for line in open(data_dir / "pool_scored.jsonl", encoding="utf-8")]
arms = json.load(open(data_dir / "arms.json"))
from transformers import AutoModelForCausalLM, AutoTokenizer


for job in queue:
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    name = job["arm"]
    if name == "ref-instruct":
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", dtype=torch.float32).eval()
        log, train_sec = [], 0.0
        tag = name
    else:
        rows = [pool[i] for i in arms[name]]
        model, tok = ns["train_arm"](rows, name, job["seed"])
        tag = name + "-s" + str(job["seed"])
        log, train_sec = ns["TRAIN_LOGS"][tag]["loss"], ns["TRAIN_LOGS"][tag]["seconds"]
    t1 = time.time()
    loose, strict, stopped, texts = ns["ifeval_scores"](model, tok)
    t2 = time.time()
    del model
    torch.cuda.empty_cache()
    rec = {"arm": name, "seed": job.get("seed"), "trainSec": round(train_sec, 1), "ifevalSec": round(t2 - t1, 1),
           "log": log, "ifevalLoose": loose.tolist(), "ifevalStrict": strict.tolist(), "ifevalStopped": float(stopped),
           "ifevalTexts": texts, "peakMemGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "wallSec": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0)}
    with open(out_dir / (tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    print(tag, "IFEval loose", round(float(loose.mean()), 3), "strict", round(float(strict.mean()), 3),
          "wall", rec["wallSec"], flush=True)

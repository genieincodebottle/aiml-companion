"""Timing figures for the lltTinyStories page, T1 (laptop CPU). Run on an IDLE machine.

python measure_timing.py
Runs the learner notebook's cells in order on the CPU (the full T1 config, 1,500 steps), with a fresh
token cache, then runs the data cell again to time a cache hit. Prints one JSON line:
  t1_tokenizer_s    tokenizer training on 300,000 stories
  t1_encode_s       encoding 200,000 training stories plus the validation split
  t1_cache_load_s   the data cell again, ids loaded from the .npy cache
  t1_train_min      the training cell, evals and samples included
  t1_tok_per_s      training tokens per second over the training cell
  t1_total_min      every cell, first run, after the dataset is in the Hugging Face cache
Env: LLT_THREADS (default 8, the notebook's own cap).

python measure_timing.py --from-run <run.json>
Prints t1_train_min and t1_total_min from a run file written by run_cells.py (its trainSec and
wallClockSec) instead of running again, with "derived_from" naming the file. Use it only for a run
made on a quiet machine.
Run with the .venv-latest interpreter (python 3.12, torch 2.15.0, tokenizers 0.23.3, datasets 5.1.0).
"""
import contextlib, io, json, os, re, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if len(sys.argv) > 2 and sys.argv[1] == "--from-run":
    r = json.load(open(sys.argv[2], encoding="utf-8"))
    print(json.dumps({"t1_train_min": round(r["trainSec"] / 60, 1), "t1_total_min": round(r["wallClockSec"] / 60, 1),
                      "derived_from": os.path.basename(sys.argv[2]), "threads": r["hardware"].get("torchThreads")}))
    sys.exit(0)

from cells_lab import CELLS

threads = int(os.environ.get("LLT_THREADS", "8"))
cache = tempfile.mkdtemp(prefix="llt_ts_cache_")
os.environ["LLT_CACHE"] = cache
os.environ.pop("LLT_STEPS", None)
ns = {"__name__": "__main__"}
times, outs = {}, {}
t_all = time.time()
code = [(i, s) for i, (k, s) in enumerate(CELLS) if k == "code"]
for i, src in code:
    src = src.replace("torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))", "torch.set_num_threads(" + str(threads) + ")")
    buf = io.StringIO(); t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "cell" + str(i), "exec"), ns)
    times[i], outs[i] = time.time() - t, buf.getvalue()
total = time.time() - t_all
data_cell = next(i for i, s in code if "Step 3. Tokenizer" in s)
train_cell = next(i for i, s in code if "Step 6. Train" in s)
tok_s = float(re.search(r"tokenizer trained in ([0-9.]+) s", outs[data_cell]).group(1))
enc_s = float(re.search(r"encoded [0-9]+ train stories in ([0-9.]+) s", outs[data_cell]).group(1))
t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(dict(code)[data_cell], "data-again", "exec"), ns)
cache_s = time.time() - t
cfg = ns["CFG"]
print(json.dumps({"t1_tokenizer_s": round(tok_s, 1), "t1_encode_s": round(enc_s, 1), "t1_cache_load_s": round(cache_s, 1),
                  "t1_train_min": round(ns["TRAIN_SEC"] / 60, 1),
                  "t1_tok_per_s": int(cfg["steps"] * cfg["batch"] * cfg["ctx"] / ns["TRAIN_SEC"]),
                  "t1_total_min": round(total / 60, 1), "threads": threads, "cpu_count": os.cpu_count()}))
shutil.rmtree(cache, ignore_errors=True)

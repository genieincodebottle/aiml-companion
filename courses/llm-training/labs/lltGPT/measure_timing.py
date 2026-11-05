"""Timing figures for the lltGPT page. Run on an IDLE machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text. PyTorch default thread count
(what a learner gets), CPU only, Hugging Face files already in the local cache.
  steps2to7_min       steps 2 to 7 in order (the quick path), minutes
  step7_cell_s        step 7 cell: load HF GPT-2 124M, load it into our build, compare, 12 greedy tokens
  step8_data_s        step 8 data cell: read TinyStories valid, train the 2,048-token BPE, encode
  toggle_run_min      one train() call, modern variant, seed 0, 600 steps, minutes
  step8b_cell_min     the step 8 experiment cell as shipped (6 variants, seed 0), minutes
  notebook_total_min  every step cell from step 2 to step 8b run in order in one namespace, minutes
The page quotes steps2to7_min, notebook_total_min, toggle_run_min and step8b_cell_min. All four are
laptop-CPU timings; the page quotes no T4 timing (the per-run T4 wall-clock of the step 8 table is in
each lltGPT-t2-*.json, wallClockSec, from Kaggle kernels llt-gpt-toggles and llt-gpt-lr).
Run with the policy venv:  ..\\..\\.venv-latest\\Scripts\\python.exe measure_timing.py
Needs the Hugging Face cache for openai-community/gpt2 and roneneldan/TinyStories (one earlier run fetches both).
"""
import contextlib
import io
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["step2", "step3", "step4", "step5", "step6", "step6b", "step7", "step8", "step8b"]


def main():
    ns = {"__name__": "__main__"}
    out, per = {}, {}
    t_all = time.perf_counter()
    for name in ORDER:
        src = (HERE / "steps" / f"{name}.py").read_text(encoding="utf-8")
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, name, "exec"), ns)
        per[name] = time.perf_counter() - t
    out["steps2to7_min"] = round(sum(per[k] for k in ORDER[:7]) / 60, 1)
    out["notebook_total_min"] = round((time.perf_counter() - t_all) / 60, 1)
    out["step7_cell_s"] = round(per["step7"])
    out["step8_data_s"] = round(per["step8"])
    out["step8b_cell_min"] = round(per["step8b"] / 60, 1)
    t = time.perf_counter()
    ns["train"](ns["TOGGLE"], steps=600, seed=0)
    out["toggle_run_min"] = round((time.perf_counter() - t) / 60, 1)
    out["threads"] = ns["torch"].get_num_threads()
    out["torch"] = ns["torch"].__version__
    import platform
    import tokenizers
    import transformers
    out["python"], out["transformers"], out["tokenizers"] = platform.python_version(), transformers.__version__, tokenizers.__version__
    print(json.dumps(out))


if __name__ == "__main__":
    main()

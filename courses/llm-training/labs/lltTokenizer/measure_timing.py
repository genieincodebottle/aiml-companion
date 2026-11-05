"""Timing figures for the lltTokenizer page. Run on an idle machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text:
  cpython_step2_train_s   browser/step2.py training loop (100 merges on 1,054 bytes), CPython, median of 5
  cpython_step4_cell_s    browser/step4.py whole cell (train 40 merges, encode, 1,000 round trips), CPython, median of 5
  ours_train_1mb_s        bpe.train on data.train_slice(250_000) (1 MB) to 1,024 tokens
  hf_train_1mb_s          Hugging Face tokenizers on the same lines, same size
  train_speedup           ours_train_1mb_s / hf_train_1mb_s, rounded to a whole number
  ours_encode_cache_s     bpe.encode with GPT-2's merges and a piece cache, FLORES devtest en+hi+es lines
  ours_encode_nocache_s   the same with no cache
  hf_encode_batch_s       tokenizers encode_batch on the same lines
  encode_speedup          ours_encode_cache_s / hf_encode_batch_s, one decimal
  regex_on_train_s        tokenizers, 28 MB, vocab 8,192, GPT-2 rule on
  regex_off_train_s       the same with use_regex=False
  notebook_cells_min      run_steps.py cells 5 to 10 in one namespace, minutes (data already cached)
  pyodide_load_s, pyodide_step2_ms, pyodide_step4_ms
                          only when LLT_PYODIDE_DIR names a folder with node_modules/pyodide (npm i pyodide@0.27.0);
                          real Pyodide 0.27.0 under Node, median of 3 runs of each browser cell
Needs `python data.py` first. CPU only.
"""
import contextlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpe  # noqa: E402
import data  # noqa: E402


def median_time(fn, n):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def run_cell(path, ns=None):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns if ns is not None else {"__name__": "__main__"})


def main():
    # LLT_TIMING_SMOKE=1 checks the script end to end on small inputs; its numbers are not for the page.
    smoke = os.environ.get("LLT_TIMING_SMOKE") == "1"
    out = {"smoke": True} if smoke else {}
    step2 = (HERE / "browser" / "step2.py").read_text(encoding="utf-8")
    text = re.search(r'text = """(.*?)"""', step2, re.S).group(1)
    data_ids = list(text.encode("utf-8"))
    out["cpython_step2_train_s"] = round(median_time(lambda: bpe.train(text, 256 + 100, use_pretokenize=False), 5), 3)
    out["cpython_step4_cell_s"] = round(median_time(lambda: run_cell(HERE / "browser" / "step4.py"), 5), 3)
    assert len(data_ids) == 1054

    sl = data.train_slice(10_000 if smoke else 250_000)
    t = time.perf_counter(); bpe.train(sl, 300 if smoke else 1024); out["ours_train_1mb_s"] = round(time.perf_counter() - t, 1)
    from measure import hf_bpe, hf_load
    _, sec = hf_bpe(300 if smoke else 1024, sl.splitlines()); out["hf_train_1mb_s"] = round(sec, 2)
    out["train_speedup"] = round(out["ours_train_1mb_s"] / out["hf_train_1mb_s"])

    from huggingface_hub import hf_hub_download
    spec = json.loads(Path(hf_hub_download("gpt2", "tokenizer.json")).read_text(encoding="utf-8"))
    merges, _ = bpe.merges_from_hf([m.split(" ") if isinstance(m, str) else m for m in spec["model"]["merges"]])
    gpt2 = hf_load("gpt2")
    ho = data.heldout()
    lines = [x for k in ("en", "hi", "es") for x in ho[k].splitlines()]
    t = time.perf_counter(); cache = {}; [bpe.encode(x, merges, cache=cache) for x in lines]
    out["ours_encode_cache_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter(); [bpe.encode(x, merges) for x in lines]; out["ours_encode_nocache_s"] = round(time.perf_counter() - t, 2)
    gpt2.encode_batch(lines[:10])
    t = time.perf_counter(); gpt2.encode_batch(lines); out["hf_encode_batch_s"] = round(time.perf_counter() - t, 2)
    out["encode_speedup"] = round(out["ours_encode_cache_s"] / out["hf_encode_batch_s"], 1)

    tt = {k: v[:200_000] for k, v in data.train_texts().items()} if smoke else data.train_texts()
    _, sec = hf_bpe(8192, data.lines(tt.values())); out["regex_on_train_s"] = round(sec)
    _, sec = hf_bpe(8192, data.lines(tt.values()), pre="noregex"); out["regex_off_train_s"] = round(sec)

    shared = {"__name__": "__main__"}
    t = time.perf_counter()
    for n in (() if smoke else (5, 6, 7, 8, 9, 10)):
        run_cell(HERE / "steps" / f"step{n}.py", shared)
    out["notebook_cells_min"] = round((time.perf_counter() - t) / 60, 1)

    pdir = os.environ.get("LLT_PYODIDE_DIR")
    if pdir and (Path(pdir) / "node_modules" / "pyodide").exists():
        shutil.copy(HERE / "pyodide_time.mjs", Path(pdir) / "pyodide_time.mjs")
        subprocess.run(["node", "pyodide_time.mjs", str(HERE)], cwd=pdir, check=True, capture_output=True)
        rec = json.loads((HERE.parent.parent / "runs" / "lltTokenizer" / "lltTokenizer-t0-pyodide.json").read_text(encoding="utf-8"))
        out["pyodide_load_s"] = round(rec["evals"]["pyodideLoadMs"] / 1000, 1)
        out["pyodide_step2_ms"] = rec["evals"]["steps"]["step2"]["medianMs"]
        out["pyodide_step4_ms"] = rec["evals"]["steps"]["step4"]["medianMs"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()

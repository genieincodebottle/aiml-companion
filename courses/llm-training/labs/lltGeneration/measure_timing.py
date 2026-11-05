"""Timing figures for the lltGeneration page. Run on an IDLE machine; prints ONE JSON line.

Needs out/gen-model.pt from train_small.py. CPU, float32, torch.set_num_threads(4), greedy decoding,
prompt = measure.PROMPT (14 tokens with out/tokenizer.json). Every key fills a [[TIME:key]] placeholder:

  nocache_s_<N>     generate N new tokens with no cache (full forward over the whole sequence every step)
  cache_s_<N>       the same N tokens with the preallocated KV cache (prefill once, then one token per step)
  speedup_<N>       nocache_s_<N> / cache_s_<N>, one decimal
                    N in 64, 128, 256, 512, 1024; median of 3 runs, 1 run for nocache at 1024
  prefill_ms_256    one forward over a 256-token prompt that fills the cache, median of 5
  decode_ms         mean time of one cached decode step after that 256-token prefill (64 steps), median of 5
  decode_tok_s      1000 / decode_ms, whole number
  prefill_tok_s     256 / prefill_ms_256 * 1000, whole number
  train_min         the notebook's training cell (steps/nb_train.py: data cached, tokenizer, 1,500 steps), minutes
  notebook_min      every notebook cell from steps/ in one process (run_steps.py --train), minutes, one decimal
                    (both skipped with --quick)

Run with the T1 environment: .venv-latest (python 3.12, torch 2.15.0, transformers 5.22.1, tokenizers 0.23.3).
    ../../.venv-latest/Scripts/python.exe measure_timing.py            # full
    ../../.venv-latest/Scripts/python.exe measure_timing.py --quick    # generation timings only
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure import PROMPT, generate  # noqa: E402
from model import load  # noqa: E402


def med(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    from tokenizers import Tokenizer
    model, _ = load(os.path.join(HERE, "out", "gen-model.pt"))
    tok = Tokenizer.from_file(os.path.join(HERE, "out", "tokenizer.json"))
    ids = tok.encode(PROMPT).ids
    out = {"promptTokens": len(ids), "threads": 4}
    with torch.no_grad():
        generate(model, ids, 16, temperature=0)  # warm-up
        for n in (64, 128, 256, 512, 1024):
            c = med(lambda: generate(model, ids, n, temperature=0, cache=True), 3)
            u = med(lambda: generate(model, ids, n, temperature=0, cache=False), 1 if n == 1024 else 3)
            out[f"cache_s_{n}"] = round(c, 3)
            out[f"nocache_s_{n}"] = round(u, 3)
            out[f"speedup_{n}"] = round(u / c, 1)
        prompt = torch.randint(1, 2048, (1, 256), generator=torch.Generator().manual_seed(0))

        def prefill():
            c = model.new_cache(1, 256 + 64)
            model(prompt, c)
            return c

        out["prefill_ms_256"] = round(med(prefill, 5) * 1000, 2)

        def decode():
            c = prefill()
            nxt = torch.tensor([[5]])
            t = time.perf_counter()
            for _ in range(64):
                model(nxt, c)
            return (time.perf_counter() - t) / 64

        out["decode_ms"] = round(statistics.median([decode() for _ in range(5)]) * 1000, 3)
    out["decode_tok_s"] = int(1000 / out["decode_ms"] + 0.5)
    out["prefill_tok_s"] = int(256 / out["prefill_ms_256"] * 1000 + 0.5)
    if not args.quick:
        # The notebook's cells (steps/) end to end in one process, its CPU training cell included.
        t = time.perf_counter()
        res = subprocess.run([sys.executable, "-u", "run_steps.py", "--train"], cwd=HERE, check=True,
                             capture_output=True, text=True)
        out["notebook_min"] = round((time.perf_counter() - t) / 60, 1)
        m = re.search(r"===== nb_train.py [(]([0-9.]+) s[)]", res.stdout)
        out["train_min"] = round(float(m.group(1)) / 60, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

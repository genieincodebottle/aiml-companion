"""Induction and ICL scores on the FINAL checkpoints of the lltTinyStories sweep (5 models, 3,000 steps each).

Checkpoints: kaggle/out/llt-tinystories-sweep/*-final-fp16.pt (float16 weights, loaded into float32 here),
with that sweep's own tokenizer.json. One point per model, so this compares models, not training steps.

Probes (as in ts_induction.py and ts_repeat_text.py):
  random tokens   n = 128 random tokens (never <|eot|>) repeated twice, 16 sequences
  real text       16 windows of n = 100 validation tokens, each followed by itself
  ICL score       loss at token 500 minus loss at token 50 over the first 32 validation windows of 512 tokens

    python ts_sweep_induction.py --threads 2
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from induction import attention_maps, induction_scores
from model import GPT
from runlog import write_run

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.normpath(os.path.join(HERE, "..", "..", "kaggle", "out", "llt-tinystories-sweep"))
RUNS = ["size3m-s0", "baseline-s1", "deep-s0", "wide-s0", "size30m-s0"]


def per_head(maps, n):
    dst = torch.arange(n, 2 * n)
    scores = [m[:, :, dst, dst - n + 1].mean(dim=(0, 2)).tolist() for m in maps]
    flat = sorted(((s, li, hi) for li, layer in enumerate(scores) for hi, s in enumerate(layer)), reverse=True)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(SWEEP, "tokenizer.json"))
    eot = tok.token_to_id("<|eot|>")
    texts = load_dataset("roneneldan/TinyStories", split="validation")["text"][:2000]
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    val = np.array(ids, dtype=np.int64)
    T, W = 512, 32
    xw = torch.tensor(np.stack([val[s:s + T] for s in range(0, W * T, T)]))
    yw = torch.tensor(np.stack([val[s + 1:s + T + 1] for s in range(0, W * T, T)]))
    rt = torch.tensor(np.stack([val[s:s + 100] for s in range(0, 16000, 1000)]))
    rt = torch.cat([rt, rt], 1)
    rows = []
    t0 = time.time()
    for name in RUNS:
        ck = torch.load(os.path.join(SWEEP, f"lltTinyStories-T2-{name}-final-fp16.pt"), map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = GPT(c["vocab"], c["d"], c["layers"], c["heads"], c["ctx"])
        m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        lo, hi = (1, c["vocab"]) if eot == 0 else (0, c["vocab"])
        with torch.no_grad():
            scores, first, second, _ = induction_scores(m, lo, hi, n=128, batch=16)
            flat_r = sorted(((s, li, hh) for li, layer in enumerate(scores) for hh, s in enumerate(layer)), reverse=True)
            lg = m(rt)
            tl = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), rt[:, 1:].reshape(-1), reduction="none").view(16, -1)
            flat_t = per_head(attention_maps(m, rt), 100)
            pos = torch.zeros(T)
            for b in range(0, W, 8):
                lw = m(xw[b:b + 8])
                pos += F.cross_entropy(lw.view(-1, lw.size(-1)), yw[b:b + 8].reshape(-1), reduction="none").view(-1, T).sum(0)
            pos /= W
        row = {"run": "lltTinyStories-T2-" + name, "params": params, "d": c["d"], "layers": c["layers"], "heads": c["heads"],
               "randomTokens": {"lossFirstCopy": round(first, 4), "lossSecondCopy": round(second, 4),
                                "inductionMax": round(flat_r[0][0], 4), "head": f"layer {flat_r[0][1]} head {flat_r[0][2]}",
                                "headsAbove0p4": sum(1 for s, _, _ in flat_r if s > 0.4)},
               "realText": {"lossFirstCopy": round(tl[:, :99].mean().item(), 4), "lossSecondCopy": round(tl[:, 100:].mean().item(), 4),
                            "inductionMax": round(flat_t[0][0], 4), "head": f"layer {flat_t[0][1]} head {flat_t[0][2]}",
                            "headsAbove0p4": sum(1 for s, _, _ in flat_t if s > 0.4)},
               "iclScore_500_minus_50": round((pos[499] - pos[49]).item(), 4)}
        rows.append(row)
        print(row, flush=True)
    write_run("lltGeneration-t1-induction-tinystories-sweep", "T1", "research:induction-tinystories-sweep",
              {"checkpoints": "kaggle/out/llt-tinystories-sweep final fp16 checkpoints, 3,000 steps each, loaded as float32",
               "tokenizer": "the sweep's tokenizer.json", "iclWindows": W, "dtype": "float32"},
              0, rows, wall_clock_sec=round(time.time() - t0, 1),
              notes="Final checkpoints only, one per model, so this compares models and cannot date a phase change.")


if __name__ == "__main__":
    main()

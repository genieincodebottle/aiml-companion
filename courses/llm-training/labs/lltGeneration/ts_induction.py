"""Research extension on our TinyStories checkpoints: induction score and ICL score at each saved step.

Checkpoints: kaggle/out/llt-tinystories-pretrain-test/ckpt/step{0,1000,2000,3000}.pt, written by
kaggle/cells_tinystories.py (run lltTinyStories-T2-baseline-s0, one T4, 98.3M training tokens).
model.GPT has the same parameter names and the same forward pass, so the state dicts load as they are.

Scores, as in induction.py
  induction score   per head, repeated random tokens (n = 128 drawn from the vocabulary minus <|eot|>,
                    repeated twice, 16 sequences), mean attention from token i of the second copy to i - n + 1.
  ICL score         Olsson et al. 2022 definition, loss at token 500 minus loss at token 50, averaged over
                    the first 160 validation windows of 512 tokens (the windows the Kaggle run evaluated).

Also re-measures val loss on the Kaggle run's first 640 windows in float32 on CPU, as a check that the
model class reproduces the logged value (logged under fp16 autocast on a T4).

    python ts_induction.py --threads 4
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from induction import induction_scores
from model import GPT
from runlog import write_run

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.normpath(os.path.join(HERE, "..", "..", "kaggle", "out", "llt-tinystories-pretrain-test", "ckpt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(CKPT, "tokenizer.json"))
    eot = tok.token_to_id("<|eot|>")
    val_text = load_dataset("roneneldan/TinyStories", split="validation")["text"]
    ids = []
    for enc in tok.encode_batch(val_text):
        ids.extend(enc.ids)
        ids.append(eot)
    val = np.array(ids, dtype=np.int64)
    T = 512
    windows = [s for s in range(0, 640 * T, T) if s + T + 1 <= val.size][:640]
    x_all = torch.tensor(np.stack([val[s:s + T] for s in windows]))
    y_all = torch.tensor(np.stack([val[s + 1:s + T + 1] for s in windows]))

    metrics = []
    t0 = time.time()
    for step in (0, 1000, 2000, 3000):
        ck = torch.load(os.path.join(CKPT, f"step{step}.pt"), map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = GPT(c["vocab"], c["d"], c["layers"], c["heads"], c["ctx"])
        m.load_state_dict(ck["model"])
        m.eval()
        nll, n = 0.0, 0
        pos_loss = torch.zeros(T)
        with torch.no_grad():
            for b in range(0, 640, 16):
                lg = m(x_all[b:b + 16])
                tl = F.cross_entropy(lg.view(-1, lg.size(-1)), y_all[b:b + 16].reshape(-1), reduction="none").view(-1, T)
                nll += tl.sum().item()
                n += tl.numel()
                if b < 160:
                    pos_loss += tl.sum(0)
        pos_loss /= 160
        icl = (pos_loss[499] - pos_loss[49]).item()
        lo, hi = (1, c["vocab"]) if eot == 0 else (0, eot) if eot == c["vocab"] - 1 else (0, c["vocab"])
        scores, first, second, _ = induction_scores(m, lo, hi, n=128, batch=16)
        flat = [(s, li, hi) for li, layer in enumerate(scores) for hi, s in enumerate(layer)]
        flat.sort(reverse=True)
        rec = {"step": step, "tokens": step * c["batch"] * c["ctx"], "valLossCpuFp32": round(nll / n, 4),
               "iclScore_500_minus_50": round(icl, 4),
               "lossAtToken": {str(k + 1): round(pos_loss[k].item(), 4) for k in (0, 9, 49, 99, 199, 499)},
               "repeatLossFirstCopy": round(first, 4), "repeatLossSecondCopy": round(second, 4),
               "inductionMax": round(flat[0][0], 4), "inductionMaxHead": f"layer {flat[0][1]} head {flat[0][2]}",
               "inductionTop5": [[round(s, 4), li, hi] for s, li, hi in flat[:5]],
               "headsAbove0p4": sum(1 for s, _, _ in flat if s > 0.4),
               "inductionByHead": [[round(s, 4) for s in layer] for layer in scores]}
        metrics.append(rec)
        print({k: v for k, v in rec.items() if k != "inductionByHead"}, flush=True)
    note = "eot id " + str(eot) + "; random tokens never include it"
    write_run("lltGeneration-t1-induction-tinystories", "T1", "research:induction-tinystories",
              {"checkpoints": "kaggle/out/llt-tinystories-pretrain-test/ckpt step0/1000/2000/3000 (run lltTinyStories-T2-baseline-s0)",
               "params": 11474240, "repeatN": 128, "repeatBatch": 16, "iclWindows": 160, "valWindows": 640, "dtype": "float32",
               "valData": "roneneldan/TinyStories validation, encoded with the run's tokenizer.json, stories joined with <|eot|>"},
              0, metrics, wall_clock_sec=round(time.time() - t0, 1),
              notes="Only 4 checkpoints were saved (every 1,000 steps = 32.8M tokens), so a phase change between them cannot be located. " + note)


if __name__ == "__main__":
    main()

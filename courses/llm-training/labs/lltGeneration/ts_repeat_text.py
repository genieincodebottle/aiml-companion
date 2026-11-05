"""Add an in-distribution induction probe to runs/lltGeneration/lltGeneration-t1-induction-tinystories.json.

The random-token probe in ts_induction.py feeds tokens TinyStories rarely uses. This probe repeats real
validation text instead: 16 windows of n = 100 tokens from the validation stream, each followed by itself
(200 tokens). It records, per checkpoint, the loss on each copy and the per-head induction score
(attention from token i of the second copy to i - n + 1).

    python ts_repeat_text.py --threads 4
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from induction import attention_maps
from model import GPT
from runlog import RUNS_DIR

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
    val_text = load_dataset("roneneldan/TinyStories", split="validation")["text"][:2000]
    ids = []
    for enc in tok.encode_batch(val_text):
        ids.extend(enc.ids)
        ids.append(eot)
    val = np.array(ids, dtype=np.int64)
    n, B = 100, 16
    starts = [i * 1000 for i in range(B)]
    r = torch.tensor(np.stack([val[s:s + n] for s in starts]))
    x = torch.cat([r, r], 1)
    path = os.path.join(RUNS_DIR, "lltGeneration-t1-induction-tinystories.json")
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    for m_rec in rec["metrics"]:
        step = m_rec["step"]
        ck = torch.load(os.path.join(CKPT, f"step{step}.pt"), map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = GPT(c["vocab"], c["d"], c["layers"], c["heads"], c["ctx"])
        m.load_state_dict(ck["model"])
        m.eval()
        with torch.no_grad():
            lg = m(x)
            tl = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none").view(B, -1)
            maps = attention_maps(m, x)
        dst = torch.arange(n, 2 * n)
        scores = [mp[:, :, dst, dst - n + 1].mean(dim=(0, 2)).tolist() for mp in maps]
        flat = sorted(((s, li, hi) for li, layer in enumerate(scores) for hi, s in enumerate(layer)), reverse=True)
        m_rec["repeatedText"] = {"n": n, "sequences": B, "lossFirstCopy": round(tl[:, : n - 1].mean().item(), 4),
                                 "lossSecondCopy": round(tl[:, n:].mean().item(), 4),
                                 "inductionMax": round(flat[0][0], 4), "inductionMaxHead": f"layer {flat[0][1]} head {flat[0][2]}",
                                 "headsAbove0p4": sum(1 for s, _, _ in flat if s > 0.4),
                                 "inductionTop5": [[round(s, 4), li, hi] for s, li, hi in flat[:5]]}
        print(step, m_rec["repeatedText"], flush=True)
    rec["config"]["repeatedTextProbe"] = "16 windows of 100 validation tokens (starts 0, 1000, ..., 15000 in the first 2,000 stories), each followed by itself"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print("updated", path)


if __name__ == "__main__":
    main()

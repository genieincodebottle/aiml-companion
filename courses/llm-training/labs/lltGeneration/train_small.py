"""Pretrain the lab model FROM SCRATCH on CPU: a 4-layer GPT on TinyStoriesV2-GPT4-valid.txt.

Same architecture as kaggle/cells_tinystories.py (model.py), smaller: vocab 2,048 byte-level BPE,
d 192, 4 layers, 6 heads (head dim 32), context 256. Stories are split 95/5 by story, shuffle seed 0,
and the tokenizer is trained on the train part only.

Writes out/gen-model.pt, out/tokenizer.json and runs/lltGeneration/<run-id>.json.

    python train_small.py --seed 0 --threads 4
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT
from runlog import write_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CFG = dict(V=2048, d=192, L=4, H=6, H_kv=6, ctx=256, batch=16, steps=1500, lr=2e-3, warmup=100, wd=0.1, clip=1.0,
           eval_every=250, eval_windows=64)


def load_stories():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]
    random.Random(0).shuffle(stories)
    cut = int(0.95 * len(stories))
    return stories[:cut], stories[cut:]


def train_tokenizer(train):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=CFG["V"], special_tokens=["<|eot|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(train, trainer=trainer)
    return tok


def encode(tok, texts):
    eot = tok.token_to_id("<|eot|>")
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    return np.array(ids, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    train, val = load_stories()
    tok = train_tokenizer(train)
    tr, va = encode(tok, train), encode(tok, val)
    print("stories", len(train), len(val), "tokens", tr.size, va.size, flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = GPT(CFG["V"], CFG["d"], CFG["L"], CFG["H"], CFG["ctx"], CFG["H_kv"])
    params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=CFG["lr"], betas=(0.9, 0.95))
    T, B = CFG["ctx"], CFG["batch"]
    vx = torch.tensor(np.stack([va[s:s + T] for s in range(0, CFG["eval_windows"] * T, T)]))
    vy = torch.tensor(np.stack([va[s + 1:s + T + 1] for s in range(0, CFG["eval_windows"] * T, T)]))

    def lr_at(s):
        if s < CFG["warmup"]:
            return CFG["lr"] * (s + 1) / CFG["warmup"]
        p = (s - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"])
        return CFG["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0
        for i in range(0, vx.shape[0], 16):
            lg = model(vx[i:i + 16])
            tot += F.cross_entropy(lg.view(-1, CFG["V"]), vy[i:i + 16].reshape(-1), reduction="sum").item()
        model.train()
        return tot / vy.numel()

    metrics = []
    t_train = time.time()
    for step in range(CFG["steps"] + 1):
        if step % CFG["eval_every"] == 0 or step == CFG["steps"]:
            metrics.append({"step": step, "tokens": step * B * T, "valLoss": round(evaluate(), 4), "lr": lr_at(step)})
            print(metrics[-1], flush=True)
        if step == CFG["steps"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ix = rng.integers(0, tr.size - T - 1, B)
        x = torch.tensor(np.stack([tr[i:i + T] for i in ix]))
        y = torch.tensor(np.stack([tr[i + 1:i + T + 1] for i in ix]))
        loss = F.cross_entropy(model(x).view(-1, CFG["V"]), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        if step % 250 == 0:
            metrics[-1]["trainLoss"] = round(loss.item(), 4)
    train_sec = time.time() - t_train
    os.makedirs(OUT, exist_ok=True)
    cfg = dict(model.cfg)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": CFG["steps"]}, os.path.join(OUT, "gen-model.pt"))
    tok.save(os.path.join(OUT, "tokenizer.json"))
    run_id = args.run_id or f"lltGeneration-t1-model-s{args.seed}"
    write_run(run_id, "T1", "baseline", dict(CFG, data="roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt, 95/5 split by story, shuffle seed 0"),
              args.seed, metrics, evals={"params": params, "trainTokens": int(tr.size), "valTokens": int(va.size),
                                         "trainStories": len(train), "valStories": len(val), "lnV": round(math.log(CFG["V"]), 4)},
              wall_clock_sec=round(time.time() - t0, 1),
              notes="Timing from a shared machine; quote only the timing script's figures. trainSec " + str(round(train_sec, 1)))


if __name__ == "__main__":
    main()

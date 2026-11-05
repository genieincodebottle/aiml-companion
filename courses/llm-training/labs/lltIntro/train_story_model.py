"""Pretrain the tiny character-level story model that lltIntro step 1 runs in the browser.

FROM SCRATCH on CPU: a 4-layer GPT (d=64, 4 heads, context 128, no biases, RMSNorm, GELU,
tied input and output embeddings) on TinyStoriesV2-GPT4-valid.txt (Eldan and Li 2023),
split 95/5 by story into our own train and validation sets.

Usage (from this folder):
  python train_story_model.py --seed 0 --run-id lltIntro-t1-story-s0 --export ../../runs/lltIntro/story-model.json
  python train_story_model.py --seed 0 --init-std 1.0 --steps 600 --run-id lltIntro-t1-broken-initstd1 --label broken:init-std-1.0

Each run writes one Run Library JSON (runlog.py). --export writes the browser weights as float16,
base64-encoded, plus the vocabulary and one held-out story.
"""
import argparse
import base64
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from runlog import write_run

ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'\"-:;" + chr(10))
NORMALISE = {"“": '"', "”": '"', "’": "'", "‘": "'", "–": "-", "—": "-"}


def load_stories():
    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
    raw = open(path, encoding="utf-8").read()
    for a, b in NORMALISE.items():
        raw = raw.replace(a, b)
    stories = [s.strip() for s in raw.split("<|endoftext|>")]
    kept = [s for s in stories if s and set(s) <= ALLOWED]
    return stories, kept


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1 = nn.RMSNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d, eps=1e-5)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        q, k, v = [a.view(B, T, self.h, D // self.h).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.n2(x)), approximate="tanh"))


class TinyGPT(nn.Module):
    def __init__(self, V, d, L, h, ctx):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.nf = nn.RMSNorm(d, eps=1e-5)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.nf(x) @ self.tok.weight.T  # tied output head


def init_weights(model, std):
    for name, p in model.named_parameters():
        if p.dim() == 2:
            nn.init.normal_(p, 0.0, std)
        else:
            nn.init.ones_(p)  # RMSNorm gains


def export_browser_json(model, chars, cfg, eval_text, path):
    w = {"tok": model.tok.weight, "pos": model.pos.weight, "nf": model.nf.weight}
    for i, b in enumerate(model.blocks):
        w["n1.%d" % i] = b.n1.weight
        w["n2.%d" % i] = b.n2.weight
        # NumPy code computes x @ W, so store Linear weights as (in, out).
        w["qkv.%d" % i] = b.qkv.weight.T
        w["proj.%d" % i] = b.proj.weight.T
        w["fc.%d" % i] = b.fc.weight.T
        w["out.%d" % i] = b.out.weight.T
    weights = {}
    for k, t in w.items():
        a = t.detach().float().contiguous().numpy().astype(np.float16)
        weights[k] = {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    doc = {"format": "lltIntro story model, float16 little-endian, base64",
           "vocab": "".join(chars), "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
           "ctx": cfg["ctx"], "d_model": cfg["d_model"], "eval_text": eval_text, "weights": weights}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        json.dump(doc, f, separators=(",", ":"))
    print("exported", path, os.path.getsize(path), "bytes")


@torch.no_grad()
def eval_loss(model, data, ctx, n_batches=40, bs=64, seed=1234):
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        ix = torch.randint(len(data) - ctx - 1, (bs,), generator=g)
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return tot / n_batches


@torch.no_grad()
def sample(model, chars, prompt, n, temp, seed, ctx):
    g = torch.Generator().manual_seed(seed)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        logits = model(torch.tensor([ids[-ctx:]]))[0, -1] / temp
        ids.append(torch.multinomial(F.softmax(logits, -1), 1, generator=g).item())
    return "".join(chars[i] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t_start = time.time()
    all_stories, stories = load_stories()
    random.Random(0).shuffle(stories)  # the split is fixed (seed 0) for every run
    n_val = len(stories) // 20
    val_stories, train_stories = stories[:n_val], stories[n_val:]
    sep = chr(10) + chr(10)
    train_text, val_text = sep.join(train_stories), sep.join(val_stories)
    chars = sorted(set(train_text) | set(val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    train = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    V = len(chars)
    print("stories kept", len(stories), "of", len(all_stories), "| V =", V,
          "| train chars", len(train), "| val chars", len(val))

    cfg = {"vocab_size": V, "d_model": 64, "n_layer": 4, "n_head": 4, "ctx": 128,
           "batch_size": 64, "steps": args.steps, "lr_max": args.lr, "lr_min": args.lr / 10,
           "warmup": 100, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0,
           "init_std": args.init_std, "dataset": "roneneldan/TinyStories TinyStoriesV2-GPT4-valid.txt",
           "stories_total": len([s for s in all_stories if s]), "stories_kept": len(stories),
           "train_chars": len(train), "val_chars": len(val), "split": "95/5 by story, shuffle seed 0",
           "threads": args.threads}

    torch.manual_seed(args.seed)
    model = TinyGPT(V, cfg["d_model"], cfg["n_layer"], cfg["n_head"], cfg["ctx"])
    init_weights(model, args.init_std)
    n_params = sum(p.numel() for p in model.parameters())
    cfg["n_params"] = n_params
    print("params", n_params, "| ln(V) =", round(math.log(V), 4))

    decay = [p for p in model.parameters() if p.dim() == 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < cfg["warmup"]:
            return args.lr * (step + 1) / cfg["warmup"]
        prog = (step - cfg["warmup"]) / max(1, args.steps - cfg["warmup"])
        return cfg["lr_min"] + 0.5 * (args.lr - cfg["lr_min"]) * (1 + math.cos(math.pi * prog))

    g = torch.Generator().manual_seed(args.seed)
    metrics = []
    ctx, bs = cfg["ctx"], cfg["batch_size"]
    v0 = eval_loss(model, val, ctx)
    metrics.append({"step": 0, "tokens": 0, "valLoss": round(v0, 4), "trainLoss": None, "lr": 0.0})
    print("step 0 val", round(v0, 4))
    t_train = time.time()
    for step in range(1, args.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step - 1)
        ix = torch.randint(len(train) - ctx - 1, (bs,), generator=g)
        x = torch.stack([train[i:i + ctx] for i in ix])
        y = torch.stack([train[i + 1:i + ctx + 1] for i in ix])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
        opt.step()
        if step == 1:
            metrics.append({"step": 1, "tokens": bs * ctx, "trainLoss": round(loss.item(), 4),
                            "firstBatchLoss": round(loss.item(), 4), "gradNorm": round(gn, 3)})
        if step % args.eval_every == 0 or step == args.steps:
            vl = eval_loss(model, val, ctx)
            el = time.time() - t_train
            metrics.append({"step": step, "tokens": step * bs * ctx, "trainLoss": round(loss.item(), 4),
                            "valLoss": round(vl, 4), "gradNorm": round(gn, 3), "lr": round(lr_at(step - 1), 6),
                            "tokPerSec": round(step * bs * ctx / el, 1), "elapsedSec": round(el, 1)})
            print("step", step, "train", round(loss.item(), 4), "val", round(vl, 4), "gn", round(gn, 2),
                  "elapsed", round(el, 1))

    final_val = eval_loss(model, val, ctx, n_batches=200, seed=99)
    model.eval()
    samples = [{"step": args.steps, "prompt": p, "temperature": 0.8, "seed": 0,
                "completion": sample(model, chars, p, 300, 0.8, 0, ctx)}
               for p in ["Once upon a time", "Tom and Lily"]]
    for s in samples:
        print(repr(s["completion"]))
    wall = time.time() - t_start
    evals = {"finalValLoss200Batches": round(final_val, 4), "lnV": round(math.log(V), 4),
             "step0ValLoss": round(v0, 4), "trainSec": round(time.time() - t_train, 1)}
    print("final val (200 batches)", round(final_val, 4), "wall", round(wall, 1))

    if args.export:
        torch.save(model.state_dict(), os.path.splitext(args.export)[0] + ".pt")
        eval_story = next(s for s in val_stories if 400 <= len(s) <= 700)
        export_browser_json(model, chars, cfg, eval_story, args.export)
        evals["exportedTo"] = os.path.basename(args.export)

    write_run(args.run_id, "T1", args.label, cfg, args.seed, metrics, evals, samples, round(wall, 1))


if __name__ == "__main__":
    main()

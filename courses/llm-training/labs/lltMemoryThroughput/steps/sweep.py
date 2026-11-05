# sweep.py - one short TinyStories run per learning rate, with and without the fixes.
# Usage: python sweep.py <size> <variant> <lr> [<lr> ...]   size: 10M or 27M, variant: base, qk or z
import json, math, sys, time
import numpy as np
import torch
from tiny import GPT

SIZES = {"10M": dict(d=320, L=8, H=8), "27M": dict(d=512, L=8, H=8)}
STEPS, WARMUP, B, T = 600, 100, 32, 256
train_ids = np.load("tok_train.npy", mmap_mode="r")
val_ids = np.load("tok_val.npy", mmap_mode="r")

def batch(arr, rng):
    ix = rng.integers(0, arr.size - T - 1, B)
    x = np.stack([arr[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + T + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

@torch.no_grad()
def evaluate(model):
    model.eval()
    rng, losses = np.random.default_rng(0), []           # the same 20 validation batches every time
    for i in range(20):
        x, y = batch(val_ids, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            losses.append(model(x, y, probe=i == 0).item())
    model.train()
    return float(np.mean(losses)), model.stats

def run(size, variant, lr, seed=0):
    torch.manual_seed(seed)
    model = GPT(qk_norm=variant == "qk", **SIZES[size]).cuda()
    z = 1e-4 if variant == "z" else 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    rng, log, skipped = np.random.default_rng(seed + 1), [], 0
    for step in range(STEPS + 1):
        if step % 150 == 0:
            vl, stats = evaluate(model)
            log.append({"step": step, "valLoss": round(vl, 4), **{k: round(v, 3) for k, v in stats.items()}})
            print(size, variant, lr, log[-1], flush=True)
        if step == STEPS:
            break
        frac = (step - WARMUP) / (STEPS - WARMUP)
        lr_now = lr * (step + 1) / WARMUP if step < WARMUP else lr * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr_now
        x, y = batch(train_ids, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, y, z_loss=z)
        scaler.scale(loss).backward()
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += scaler.get_scale() < scale                # the scaler lowers its scale when it skips a step
        opt.zero_grad(set_to_none=True)
    params = sum(p.numel() for p in model.parameters())
    return {"size": size, "params": params, "variant": variant, "lr": lr, "seed": seed, "skippedSteps": int(skipped),
            "final": log[-1], "metrics": log}

if __name__ == "__main__":
    size, variant, lrs = sys.argv[1], sys.argv[2], [float(a) for a in sys.argv[3:]]
    for lr in lrs:
        t = time.time()
        r = run(size, variant, lr)
        r["wallClockSec"] = round(time.time() - t, 1)
        with open(f"p3_{size}_{variant}_{lr:g}.json", "w") as f:
            json.dump(r, f)

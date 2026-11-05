"""Induction score and in-context-learning score (research extension, Olsson et al. 2022).

Scores
  induction score   per head, the mean attention weight from each token of the second copy of a repeated
                    random sequence to the token right after that token's first occurrence
                    (destination i, source i - n + 1, for i = n .. 2n - 1). 1.0 is a perfect induction head.
  ICL score         loss at a late token minus loss at an early token, averaged over sequences.
                    Olsson et al. use token 500 minus token 50 on natural text with a 512-token context.

Verification here (main): a 2-layer attention-only model trained FROM SCRATCH on CPU on sequences of
n random tokens repeated twice. Writes runs/lltGeneration/lltGeneration-t1-induction-toy-s<seed>.json.

    python induction.py --seed 0 --threads 4
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

from model import GPT, apply_rope, rope_tables


@torch.no_grad()
def attention_maps(model, idx):
    """Attention weights of every head, recomputed from the model's own q and k. Returns [layer] of (B, H, T, T)."""
    T = idx.shape[1]
    hd = model.emb.embedding_dim // model.H
    cos, sin = rope_tables(0, T, hd, idx.device)
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    x = model.emb(idx)
    maps = []
    for b in model.blocks:
        B, _, C = x.shape
        q, k, _ = b.qkv(b.n1(x)).split([C, b.h_kv * b.hd, b.h_kv * b.hd], dim=2)
        q = apply_rope(q.view(B, T, b.h, b.hd).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, b.h_kv, b.hd).transpose(1, 2), cos, sin)
        if b.h_kv != b.h:
            k = k.repeat_interleave(b.h // b.h_kv, dim=1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(b.hd)
        maps.append(s.masked_fill(~causal, float("-inf")).softmax(-1))
        x = b(x, cos, sin)
    return maps


def train_batch(V, batch, T, g):
    """n random tokens repeated until the sequence is T long, with the period n drawn from 8..32 per sequence."""
    x = torch.empty(batch, T, dtype=torch.long)
    for b in range(batch):
        n = int(torch.randint(8, 33, (1,), generator=g))
        r = torch.randint(0, V, (n,), generator=g)
        x[b] = r.repeat(T // n + 1)[:T]
    return x


def repeated_random(vocab_lo, vocab_hi, n, batch, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(vocab_lo, vocab_hi, (batch, n), generator=g)
    return torch.cat([r, r], dim=1)


@torch.no_grad()
def induction_scores(model, vocab_lo, vocab_hi, n=32, batch=16, seed=1234):
    """Per-head induction scores, a list of layers, each a list of head scores, plus the loss on each copy."""
    idx = repeated_random(vocab_lo, vocab_hi, n, batch, seed)
    maps = attention_maps(model, idx)
    dst = torch.arange(n, 2 * n)
    src = dst - n + 1
    scores = [m[:, :, dst, src].mean(dim=(0, 2)).tolist() for m in maps]
    logits = model(idx)
    tok_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1), reduction="none").view(batch, -1)
    first = tok_loss[:, : n - 1].mean().item()     # predicting tokens 1 .. n-1, unpredictable
    second = tok_loss[:, n:].mean().item()          # predicting tokens n+1 .. 2n-1, copyable
    return scores, first, second, tok_loss.mean(0)


@torch.no_grad()
def icl_score(model, seqs, late, early):
    """Mean loss at token `late` minus mean loss at token `early` (0-based target positions), over seqs (B, T)."""
    logits = model(seqs[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seqs[:, 1:].reshape(-1), reduction="none").view(seqs.shape[0], -1)
    return (loss[:, late - 1].mean() - loss[:, early - 1].mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fixed-period", action="store_true",
                    help="train on n = 32 tokens repeated twice, always at the same offset (the flawed first design)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    CFG = dict(V=128, d=64, L=args.layers, H=4, ctx=64, n=32, batch=32, lr=1e-3, warmup=100, wd=0.0,
               steps=args.steps, data="n random tokens from 128 repeated until the sequence is 64 long, with the period n drawn from 8..32 for each sequence, fresh every step. The period varies, so a head that attends a fixed distance back cannot predict the repeats",
               probe="n = 32 random tokens repeated twice, 64 sequences")
    model = GPT(CFG["V"], CFG["d"], CFG["L"], CFG["H"], CFG["ctx"], mlp=False)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=CFG["wd"])
    g = torch.Generator().manual_seed(args.seed + 100)
    metrics = []
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            model.eval()
            scores, first, second, _ = induction_scores(model, 0, CFG["V"], CFG["n"], 64)
            seqs = repeated_random(0, CFG["V"], CFG["n"], 64, 999)
            icl = icl_score(model, seqs, late=60, early=10)
            last = scores[-1]
            metrics.append({"step": step, "tokens": step * CFG["batch"] * 2 * CFG["n"],
                            "lossFirstCopy": round(first, 4), "lossSecondCopy": round(second, 4),
                            "iclScore_60_minus_10": round(icl, 4),
                            "inductionByHead": [[round(s, 4) for s in layer] for layer in scores],
                            "inductionMaxLastLayer": round(max(last), 4)})
            print(metrics[-1]["step"], metrics[-1]["lossSecondCopy"], metrics[-1]["iclScore_60_minus_10"], metrics[-1]["inductionMaxLastLayer"], flush=True)
            model.train()
        if step == args.steps:
            break
        lr = CFG["lr"] * min(1.0, (step + 1) / CFG["warmup"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        if args.fixed_period:
            r = torch.randint(0, CFG["V"], (CFG["batch"], CFG["n"]), generator=g)
            x = torch.cat([r, r], dim=1)
        else:
            x = train_batch(CFG["V"], CFG["batch"], 64, g)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, CFG["V"]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wall = round(time.time() - t0, 1)

    # Phase change: the first eval where the best last-layer head passes 0.5, and the eval interval
    # over which the ICL score fell the most.
    cross = next((m["step"] for m in metrics if m["inductionMaxLastLayer"] >= 0.5), None)
    drops = [(metrics[i - 1]["iclScore_60_minus_10"] - metrics[i]["iclScore_60_minus_10"], metrics[i - 1]["step"], metrics[i]["step"]) for i in range(1, len(metrics))]
    big = max(drops)
    jump = [(metrics[i]["inductionMaxLastLayer"] - metrics[i - 1]["inductionMaxLastLayer"], metrics[i - 1]["step"], metrics[i]["step"]) for i in range(1, len(metrics))]
    bigj = max(jump)
    evals = {"params": params, "lnV": round(math.log(CFG["V"]), 4),
             "inductionCrosses0p5AtStep": cross,
             "largestIclDrop": {"drop": round(big[0], 4), "fromStep": big[1], "toStep": big[2]},
             "largestInductionJump": {"jump": round(bigj[0], 4), "fromStep": bigj[1], "toStep": bigj[2]},
             "final": metrics[-1]}
    print(evals)
    from runlog import write_run
    if args.fixed_period:
        CFG["data"] = "n = 32 random tokens from 128 repeated twice, always at offset 0 (length 64): a fixed period"
    run_id = args.run_id or f"lltGeneration-t1-induction-toy-L{args.layers}-s{args.seed}" + ("-fixedperiod" if args.fixed_period else "")
    label = "research:induction-toy" if args.layers > 1 else "research:induction-toy-1layer-control"
    write_run(run_id, "T1", label + (":fixed-period" if args.fixed_period else ""),
              CFG, args.seed, metrics, evals=evals, wall_clock_sec=wall,
              notes="Timing is from a shared machine with other jobs running; do not quote wallClockSec.")


if __name__ == "__main__":
    main()

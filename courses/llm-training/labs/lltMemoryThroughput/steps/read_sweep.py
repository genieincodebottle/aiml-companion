import glob, json, math

runs = [json.load(open(f)) for f in glob.glob("p3_*.json")]
table = {}
for r in runs:
    table.setdefault((r["size"], r["variant"]), {})[r["lr"]] = r
for (size, variant), by_lr in sorted(table.items()):
    lrs = sorted(by_lr)
    final = [by_lr[lr]["final"] for lr in lrs]
    start = [by_lr[lr]["metrics"][0]["valLoss"] for lr in lrs]
    # A run that diverged counts as its step-0 loss, as in Wortsman et al. 2023
    clipped = [min(f["valLoss"], s) if math.isfinite(f["valLoss"]) else s for f, s in zip(final, start)]
    best = min(clipped)
    sensitivity = sum(c - best for c in clipped) / len(clipped)
    good = [lr for lr, c in zip(lrs, clipped) if c <= best + 0.1]
    print(f"{size} {variant:4} val loss  " + "  ".join(f"{lr:g}={c:.3f}" for lr, c in zip(lrs, clipped)))
    print(f"         max attention logit  " + "  ".join(f"{f['maxAttnLogit']:.0f}" for f in final))
    print(f"         mean log Z  " + "  ".join(f"{f['meanLogZ']:.1f}" for f in final))
    print(f"         LR sensitivity {sensitivity:.3f}   within 0.1 nats of best from {min(good):g} to {max(good):g}")

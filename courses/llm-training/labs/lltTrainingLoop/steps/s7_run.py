R = dict(CPU_CFG, dropout=0.1, steps=200, eval_every=100, ckpt="resume.pt", ckpt_every=100)

_, straight = train(R)                                   # 200 steps in one go
_, first = train(dict(R, stop_at=100))                   # 100 steps, checkpoint at step 100
_, second = train(dict(R, resume=True, ckpt_every=10**9))  # new objects, load, run steps 101 to 200


def compare(a, b):
    diff = max(abs(x["trainLoss"] - y["trainLoss"]) for x, y in zip(a, b))
    return diff, b[-1]["valLoss"] - a[-1]["valLoss"]


diff, dval = compare(straight, first + second)
print("full checkpoint | max train-loss diff", diff, "| val diff at step 200", dval)

for missing in ["rng", "sampler", "scheduler", "optimizer"]:
    _, part = train(dict(R, resume=True, ckpt_every=10**9, skip=(missing,)))
    diff, dval = compare(straight, first + part)
    print(f"without {missing:<9} | max train-loss diff {diff:.4f} | val diff at step 200 {dval:+.4f}",
          f"| loss at step 101 {part[0]['trainLoss']:.4f} vs {straight[100]['trainLoss']:.4f}")

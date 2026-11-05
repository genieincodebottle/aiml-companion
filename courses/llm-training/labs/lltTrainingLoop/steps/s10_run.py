ACC = dict(CPU_CFG, micro_batch=3, accum=4)    # 4 micro-batches of 3, the same 12 sequences per update
RUNS = {
    "control, accumulate 4 x 3": ACC,
    "loss not divided by 4": dict(ACC, divide_loss=False),
    "scheduler stepped per micro-batch": dict(ACC, sched_per_micro=True),
    "learning rate 1e-1": dict(CPU_CFG, lr=1e-1),
}
for name, cfg in RUNS.items():
    _, hist = train(cfg)
    vals = {r["step"]: round(r["valLoss"], 4) for r in hist if "valLoss" in r}
    clipped = sum(r["gradNorm"] > cfg["clip"] for r in hist)
    print(f"{name:<34} val {vals} | clipped steps {clipped} | lr at step 600 {hist[599]['lr']:.2e}")

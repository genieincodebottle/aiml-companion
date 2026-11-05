T2_CFG = dict(T2, dropout=0.2, micro_batch=64, accum=1, steps=2000, warmup=100, schedule="cosine",
              wd=0.1, clip=1.0, eval_every=250)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = [0]              # one seed per rate on a free T4; [0, 1, 2] for the full sweep
print("device", device, "| dtype", pick_dtype())

results = []
if device.type != "cuda":
    print("This step needs a GPU. Use the table of our runs below it.")
else:
    for lr in LRS:
        for seed in SEEDS:
            _, hist = train(dict(T2_CFG, lr=lr, seed=seed), log_path=f"t2-lr{lr:.0e}-s{seed}.jsonl")
            results.append({"lr": lr, "seed": seed, "final": hist[-1]["valLoss"]})
            print(f"lr {lr:.0e} seed {seed} | final val {hist[-1]['valLoss']:.4f}")

torch.set_num_threads(4)
if device.type == "cpu":
    torch.use_deterministic_algorithms(True)

CPU_CFG = dict(SMALL, dropout=0.0, micro_batch=12, accum=1, steps=2000, lr=1e-3, warmup=100,
               schedule="cosine", wd=0.1, clip=1.0, seed=0, eval_every=250)
print("device", device, "| dtype", pick_dtype())
model, history = train(dict(CPU_CFG, ckpt="small-s0.pt", ckpt_every=2000), log_path="small-s0.jsonl")
for row in history:
    if "valLoss" in row:
        print("step", row["step"], "| train", round(row["trainLoss"], 4), "| val", round(row["valLoss"], 4),
              "| grad norm", round(row["gradNorm"], 3), "| lr", f"{row['lr']:.2e}")

# probe.py - largest micro-batch that fits one T4 (one full train step, no DDP).
# Usage: CUDA_VISIBLE_DEVICES=0 python probe.py <size> <ckpt 0|1> <b> [<b> ...]
import json, sys
import torch
from gpt import GPT, CONFIGS

size, ckpt, sizes = sys.argv[1], sys.argv[2] == "1", [int(b) for b in sys.argv[3:]]
cfg = CONFIGS[size]
torch.manual_seed(0)
model = GPT(**cfg, ckpt=ckpt).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
scaler = torch.amp.GradScaler("cuda")
res = {}
for b in sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randint(0, cfg["V"], (b, cfg["ctx"]), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        res[b] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except torch.cuda.OutOfMemoryError:
        res[b] = "OOM"
        opt.zero_grad(set_to_none=True)
        break
print(json.dumps({"size": size, "ckpt": ckpt, "peakGB": res,
                  "params": sum(p.numel() for p in model.parameters())}))

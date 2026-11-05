import gc
import torch
from gpt import GPT, GPTConfig

cfg = GPTConfig()

def train_step(model, opt, scaler, b, s, dtype=torch.float16):
    x = torch.randint(0, cfg.vocab, (b, s), device="cuda")
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        loss = model(x, x)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss

def measure_peak(b, s, **switches):
    """Peak allocated and reserved GB of the third training step, or None on out-of-memory."""
    model = GPT(cfg, **switches).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    result = None
    try:
        for i in range(3):
            if i == 2:
                torch.cuda.reset_peak_memory_stats()   # Adam state exists from step 1 on
            train_step(model, opt, scaler, b, s)
        torch.cuda.synchronize()
        result = (torch.cuda.max_memory_allocated() / GB, torch.cuda.max_memory_reserved() / GB)
    except torch.cuda.OutOfMemoryError:
        pass
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result

grid = []
for s in (256, 512, 1024):
    for b in (1, 2, 4, 8, 12, 16):
        pred, got = predict_peak_gb(b, s), measure_peak(b, s)
        grid.append((b, s, pred, got))
        if got is None:
            print(f"b={b:>2} s={s:>4}  predicted {pred:5.2f} GB  measured OUT OF MEMORY")
            break
        print(f"b={b:>2} s={s:>4}  predicted {pred:5.2f}  allocated {got[0]:5.2f}  "
              f"reserved {got[1]:5.2f} GB  error {pred / got[0] - 1:+.1%}")

import time

def tokens_per_sec(b, s, steps=10, dtype=torch.float16, compile=False, **switches):
    model = GPT(cfg, **switches).cuda()
    if compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    t = time.perf_counter()
    train_step(model, opt, scaler, b, s, dtype)      # the first step includes any compile time
    torch.cuda.synchronize()
    first = time.perf_counter() - t
    for _ in range(2):
        train_step(model, opt, scaler, b, s, dtype)  # warm-up
    torch.cuda.synchronize()                         # wait for queued GPU work before starting the clock
    t = time.perf_counter()
    for _ in range(steps):
        train_step(model, opt, scaler, b, s, dtype)
    torch.cuda.synchronize()                         # and again before stopping it
    tps = b * s * steps / (time.perf_counter() - t)
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return tps, first

def mfu(tps, s, peak=65e12):
    flops_per_token = 6 * N + 12 * L * d * s     # 6N for the weights, plus attention scores
    return tps * flops_per_token / peak

tps, first = tokens_per_sec(8, 1024)
print(f"b=8 s=1024  {tps:,.0f} tokens/s  first step {first:.2f} s  MFU {mfu(tps, 1024):.1%} of 65 TFLOPS")

# Timing pieces without synchronize: the clock stops when the CPU has queued the work, not when the GPU has done it
model = GPT(cfg).cuda()
x = torch.randint(0, cfg.vocab, (8, 1024), device="cuda")
for sync in (False, True):
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, x)
        if sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
    print(f"synchronize={sync!s:5}  forward {1e3 * (t1 - t0):6.1f} ms  backward {1e3 * (t2 - t1):6.1f} ms")
del model, x, loss
gc.collect()
torch.cuda.empty_cache()

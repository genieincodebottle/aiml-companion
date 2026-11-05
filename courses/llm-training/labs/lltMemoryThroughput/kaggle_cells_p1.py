import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cells_common import SETUP, src, step_cell, write_file_cell

DIAG = r'''
import gc, torch
GB = 1e9
# Memory at each phase of one step, b=8 s=1024, for the account in step 2.
def phases(**sw):
    model = GPT(cfg, **sw).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(2):
        train_step(model, opt, scaler, 8, 1024)
    torch.cuda.synchronize()
    x = torch.randint(0, cfg.vocab, (8, 1024), device="cuda")
    opt.zero_grad(set_to_none=True)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x, x)
    fwd = torch.cuda.memory_allocated(); fwd_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    scaler.scale(loss).backward()
    bwd_peak = torch.cuda.max_memory_allocated(); after_bwd = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    scaler.step(opt); scaler.update(); torch.cuda.synchronize()
    step_peak = torch.cuda.max_memory_allocated()
    r = {k: round(v / GB, 3) for k, v in dict(before=base, after_forward=fwd, forward_peak=fwd_peak, backward_peak=bwd_peak,
         after_backward=after_bwd, optimizer_peak=step_peak).items()}
    r["saved_per_token_bytes"] = round((fwd - base) / 8192)
    del model, opt, scaler, x, loss; gc.collect(); torch.cuda.empty_cache()
    return r
RUN["data"]["phases"] = {name: phases(**sw) for name, sw in variants.items()}
print(json.dumps(RUN["data"]["phases"], indent=1))

# Hidden state only (no head): bytes kept per token per block.
model = GPT(cfg).cuda()
x = torch.randint(0, cfg.vocab, (8, 1024), device="cuda")
base = torch.cuda.memory_allocated()
with torch.autocast("cuda", dtype=torch.float16):
    h = model.wte(x) + model.wpe(torch.arange(1024, device="cuda"))
    for b in model.blocks:
        h = b(h)
per_block = (torch.cuda.memory_allocated() - base) / 8192 / 12
RUN["data"]["bytes_per_token_per_block"] = round(per_block, 1)
print("bytes per token per block", round(per_block, 1), "=", round(per_block / 768, 2), "x d")
del model, x, h; gc.collect(); torch.cuda.empty_cache()

# Broken run: model.half() with AdamW. eps = 1e-8 rounds to 0 in fp16.
print("1e-8 in fp16 ->", torch.tensor(1e-8, dtype=torch.float16).item())
res = {}
for mode in ("half", "autocast"):
    torch.manual_seed(0)
    model = GPT(cfg).cuda()
    if mode == "half":
        model = model.half()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "autocast")
    x = torch.randint(0, cfg.vocab, (4, 512), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    losses, nonfinite = [], []
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=mode == "autocast"):
            loss = model(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        losses.append(round(loss.item(), 4))
        nonfinite.append(int(sum((~torch.isfinite(p)).sum().item() for p in model.parameters())))
    res[mode] = {"losses": losses, "nonfiniteParams": nonfinite}
    print(mode, losses[:6], "...", losses[-3:], "| non-finite params after steps 1-3:", nonfinite[:3])
    del model, opt; gc.collect(); torch.cuda.empty_cache()
RUN["data"]["half_vs_autocast"] = res

# Save the numbers the steps computed.
RUN["data"]["N"] = N
RUN["data"]["grid"] = [list(g[:3]) + [list(g[3]) if g[3] else None] for g in grid]
RUN["data"]["cut"] = cut
RUN["data"]["speed"] = speed
RUN["data"]["gpu"] = torch.cuda.get_device_name(0)
save("lltMemoryThroughput_p1")
'''

CELLS = [
    ("md", "# lltMemoryThroughput part 1 measurements (one T4)\n\nRuns every part 1 step of the lab page exactly as shipped and records its output."),
    ("code", "import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\n" + SETUP),
    ("code", write_file_cell("gpt.py", src("gpt.py"))),
    ("code", step_cell("check", "check_runtime.py")),
    ("code", step_cell("s01", "s01_static.py")),
    ("code", step_cell("s02", "s02_predict.py")),
    ("code", step_cell("s03", "s03_grid.py")),
    ("code", step_cell("s04", "s04_timing.py")),
    ("code", step_cell("s05", "s05_cut.py")),
    ("code", step_cell("s06", "s06_precision.py")),
    ("code", DIAG),
]

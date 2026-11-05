peak, warmup, total = 1e-3, 100, 2000
print("step   cosine     wsd")
for s in [0, 50, 99, 100, 1000, 1599, 1600, 1800, 1999]:
    print(f"{s:>4}  {lr_cosine(s, peak, warmup, total):.2e}  {lr_wsd(s, peak, warmup, total):.2e}")

# The scheduler object holds its position, so it goes into the checkpoint.
opt = torch.optim.AdamW(GPT(len(chars), **SMALL).parameters(), lr=peak)
sched = make_scheduler(opt, dict(schedule="cosine", lr=peak, warmup=warmup, steps=total))
for _ in range(150):
    opt.step()
    sched.step()
print("after 150 steps", sched.state_dict()["last_epoch"], "| lr", f"{opt.param_groups[0]['lr']:.3e}")

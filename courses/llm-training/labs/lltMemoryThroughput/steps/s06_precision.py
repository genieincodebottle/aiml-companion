speed = {}
for name, kw in [("fp32", {"dtype": torch.float32}), ("fp16 autocast", {}),
                 ("bf16 autocast", {"dtype": torch.bfloat16}), ("fp16 + torch.compile", {"compile": True})]:
    tps, first = tokens_per_sec(4, 1024, **kw)
    speed[name] = (tps, first)
    print(f"{name:22} {tps:7,.0f} tokens/s  first step {first:6.1f} s")

# fp16 without loss scaling: how many gradient entries in the blocks round to exactly zero?
torch.manual_seed(0)
model = GPT(cfg).cuda()
x = torch.randint(0, cfg.vocab, (8, 1024), device="cuda")
for scale in (1.0, 65536.0):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x, x)
    (loss * scale).backward()
    g = torch.cat([p.grad.flatten() for p in model.blocks.parameters()])
    print(f"loss scale {scale:>7.0f}  zero gradient entries {(g == 0).float().mean().item():.2%}")
print("smallest fp16 above zero:", torch.finfo(torch.float16).tiny, "normal,", 2.0 ** -24, "subnormal")
del model, x, loss, g
gc.collect()
torch.cuda.empty_cache()

@torch.no_grad()
def check_cache(model, ids, n=200):
    cache = model.new_cache(1, len(ids) + n)
    steps = [model(torch.tensor([ids]), cache)[0, -1]]         # prefill, the whole prompt at once
    toks = []
    for _ in range(n - 1):
        nxt = steps[-1].argmax().view(1, 1)
        toks.append(nxt.item())
        steps.append(model(nxt, cache)[0, -1])                  # decode, one token per call
    toks.append(steps[-1].argmax().item())
    full = model(torch.tensor([ids + toks[:-1]]))[0, len(ids) - 1:]   # no cache, one pass
    cached = torch.stack(steps)
    err = (cached - full).abs().max().item()
    print(f"{n} steps  max abs error {err:.1e}  largest logit {full.abs().max().item():.1f}")
    print("allclose:", torch.allclose(cached, full, atol=1e-4, rtol=1e-4),
          "| same greedy tokens:", toks == full.argmax(-1).tolist())

check_cache(model, ids)

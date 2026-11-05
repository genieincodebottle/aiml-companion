import time

@torch.no_grad()
def generate_cached(model, ids, n):
    cache = model.new_cache(1, len(ids) + n)
    logits = model(torch.tensor([ids]), cache)[:, -1]          # prefill
    out = []
    for _ in range(n):
        nxt = logits.argmax(-1, keepdim=True)
        out.append(nxt.item())
        logits = model(nxt, cache)[:, -1]                      # decode
    return out

for n in [64, 128, 256, 512, 1024]:
    t = time.perf_counter(); a = generate_nocache(model, ids, n, temperature=0); slow = time.perf_counter() - t
    t = time.perf_counter(); b = generate_cached(model, ids, n); fast = time.perf_counter() - t
    print(f"{n:5d} tokens  no cache {slow:7.2f} s  cache {fast:6.2f} s  speedup {slow / fast:5.1f}x  same tokens {a == b}")

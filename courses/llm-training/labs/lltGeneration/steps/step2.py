def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    z = logits / temperature
    if top_k:
        kth = torch.topk(z, top_k).values[..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    if top_p < 1.0:
        srt, order = torch.sort(z, descending=True)
        p = srt.softmax(-1)
        drop = p.cumsum(-1) - p >= top_p          # the tokens ranked above already reach top_p
        z = z.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return z.log_softmax(-1)

@torch.no_grad()
def generate_nocache(model, ids, n, temperature=1.0, top_k=0, top_p=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids])
    for _ in range(n):
        logits = model(x)[:, -1]                  # the whole sequence again, every step
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(filter_logits(logits, temperature, top_k, top_p).exp(), 1, generator=g)
        x = torch.cat([x, nxt], 1)
    return x[0, len(ids):].tolist()

prompt = "Once upon a time, there was a little girl named Lily. She"
ids = tok.encode(prompt).ids
print(len(ids), "prompt tokens")
print(prompt + tok.decode(generate_nocache(model, ids, 40, temperature=0.8, top_k=50)))

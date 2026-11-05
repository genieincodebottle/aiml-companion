with torch.no_grad():
    logits = model(torch.tensor([ids]))[0, -1]
for T in [0.5, 0.8, 1.0, 1.5]:
    lp = filter_logits(logits, T)
    p = lp.exp()
    entropy = -(p * lp).sum().item()
    kept = int((filter_logits(logits, T, top_p=0.9) > float("-inf")).sum())
    print(f"T {T}  entropy {entropy:.2f} nats  top token {p.max().item():.2f}  top-p 0.9 keeps {kept} of {len(p)}")

def repeated_4grams(toks):
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1 - len(set(grams)) / len(grams)

for name, kw in [("greedy", dict(temperature=0)), ("top-p 0.9", dict(top_p=0.9))]:
    out = generate_nocache(model, ids, 200, seed=0, **kw)
    print(f"{name:10} repeated 4-grams {repeated_4grams(out):.2f}   {tok.decode(out[:24])!r}")

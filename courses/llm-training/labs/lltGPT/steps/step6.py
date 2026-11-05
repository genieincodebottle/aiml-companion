def numel(model):
    return sum(p.numel() for p in model.parameters())   # a tied tensor counts once

def approx_params(cfg):
    return 12 * cfg.layers * cfg.d ** 2 + cfg.vocab * cfg.d

TOGGLE = replace(MODERN, vocab=2048, ctx=128, d=192, layers=4, heads=6)   # the step 8 experiment
for name, cfg in [("GPT-2 124M", GPT2), ("modern, GPT-2 size", MODERN), ("TinyStories model", TINY), ("step 8 model", TOGGLE),
                  ("step 8, 50k vocab", replace(TOGGLE, vocab=50257))]:
    n, a, emb = numel(GPT(cfg)), approx_params(cfg), cfg.vocab * cfg.d
    print(f"{name:18}  sum(numel) {n:>11,}  12Ld^2 + Vd {a:>11,}  gap {(n - a) / n:+.2%}  embedding {emb / n:.0%}")

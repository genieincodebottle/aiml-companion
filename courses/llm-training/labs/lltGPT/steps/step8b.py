VARIANTS = {
    "modern": TOGGLE,
    "learned positions": replace(TOGGLE, pos="learned"),
    "GELU MLP": replace(TOGGLE, mlp="gelu"),
    "LayerNorm": replace(TOGGLE, norm="layernorm"),
    "post-norm": replace(TOGGLE, prenorm=False),
    "gpt2 config": replace(TOGGLE, norm="layernorm", mlp="gelu", pos="learned", bias=True),
}
SEEDS = [0]   # our table used seeds 0 and 1
for name, cfg in VARIANTS.items():
    for seed in SEEDS:
        model, curve = train(cfg, seed=seed)
        print(f"{name:18} seed {seed}  parameters {numel(model):,}  val loss at step 600 {curve[-1][1]:.4f}")

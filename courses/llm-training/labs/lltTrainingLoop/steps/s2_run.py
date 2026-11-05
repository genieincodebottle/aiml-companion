SMALL = dict(n_layer=4, n_head=4, d=128, block=64)
T2 = dict(n_layer=6, n_head=6, d=384, block=256)

for name, size in [("small (CPU)", SMALL), ("10M (T4)", T2)]:
    model = GPT(len(chars), **size)
    groups = param_groups(model, weight_decay=0.1)
    count = lambda g: sum(p.numel() for p in g["params"])
    print(name, "| parameters", sum(p.numel() for p in model.parameters()),
          "| decay", len(groups[0]["params"]), "tensors", count(groups[0]),
          "| no decay", len(groups[1]["params"]), "tensors", count(groups[1]))

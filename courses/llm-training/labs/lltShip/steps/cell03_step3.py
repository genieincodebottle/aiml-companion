def task_arithmetic(base, deltas, lams):
    # base + lambda_1 tau_1 + lambda_2 tau_2 + ...
    out = base.float().clone()
    for d, lam in zip(deltas, lams):
        out += lam * d
    return out

def linear(models, weights):
    # A weighted average of whole models, a "model soup". With a shared base it is task arithmetic with lambda_i = w_i / sum(w)
    return sum(w * m.float() for m, w in zip(models, weights)) / sum(weights)

def trim(d, density):
    # TIES step 1: keep the largest-magnitude share density of the entries, zero the rest
    if density >= 1:
        return d
    k = max(1, int(density * d.numel()))
    keep = torch.argsort(d.abs().reshape(-1), descending=True)[:k]
    mask = torch.zeros(d.numel(), dtype=d.dtype)
    mask[keep] = 1
    return d * mask.reshape(d.shape)

def ties(base, deltas, density, lam=1.0, weights=None):
    weights = weights or [1.0] * len(deltas)
    w = torch.tensor(weights).reshape(-1, *[1] * base.dim())
    t = torch.stack([trim(d, density) for d in deltas]) * w      # trimmed, weighted task vectors
    sign = torch.where(t.sum(0) >= 0, 1.0, -1.0)                  # step 2: elect the sign with the larger total mass
    agree = t.sign() == sign                                      # entries that agree; a trimmed zero never agrees
    count = (w * agree).sum(0)
    merged = (t * agree).sum(0) / torch.where(count == 0, 1.0, count)   # step 3: disjoint mean over the agreeing ones
    return base.float() + lam * merged

def dare(d, p, gen):
    # DARE: drop each entry with probability p, scale the survivors by 1 / (1 - p) so the expected value is unchanged
    keep = torch.bernoulli(torch.full(d.shape, 1 - p), generator=gen)
    return d * keep / (1 - p)

# Check every function on four numbers you can follow by hand
b = torch.zeros(4)
t1, t2 = torch.tensor([0.5, -0.2, 0.1, 0.0]), torch.tensor([0.3, 0.4, -0.6, 0.05])
r4 = lambda v: [round(x, 4) for x in v.tolist()]
print("task arithmetic, lambda 1 ", r4(task_arithmetic(b, [t1, t2], [1, 1])))
print("linear, equal weights     ", r4(linear([b + t1, b + t2], [1, 1])))
print("trim to density 0.5       ", r4(trim(t1, 0.5)), r4(trim(t2, 0.5)))
print("TIES, density 0.5         ", r4(ties(b, [t1, t2], 0.5)))
big = torch.full((100000,), 0.01)
dropped = dare(big, 0.9, torch.Generator().manual_seed(0))
print("DARE p 0.9 keeps", int((dropped != 0).sum()), "of 100000 | mean before", round(float(big.mean()), 5),
      "after", round(float(dropped.mean()), 5))
assert torch.allclose(task_arithmetic(b, [t1, t2], [0.5, 0.5]), linear([b + t1, b + t2], [1, 1]))
assert torch.allclose(ties(b, [t1, t2], 0.5), torch.tensor([0.5, 0.4, -0.6, 0.0]))

# One real tensor: what TIES keeps from each fine-tune
k = "model.layers.13.mlp.down_proj.weight"
d = [tau("sft", k), tau("hindi", k)]
t = torch.stack([trim(x, 0.2) for x in d])
sign = torch.where(t.sum(0) >= 0, 1.0, -1.0)
for i, n in enumerate(("sft", "hindi")):
    kept, agree = t[i] != 0, t[i].sign() == sign
    print(n.ljust(6), "| kept by trim", round(100 * float(kept.float().mean()), 1), "% | of those, lost the sign vote",
          round(100 * float((kept & ~agree).sum() / kept.sum()), 1), "% | share of its squared norm kept after the vote",
          round(float((t[i] * agree).pow(2).sum() / d[i].pow(2).sum()), 3))

def merge_model(method, lams=(1.0, 1.0), density=1.0, p=0.0, seed=0, dtype=torch.float16):
    # Merge every tensor of the two fine-tunes into the base, one tensor at a time, on the CPU
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k in W["base"]:
        base, deltas = W["base"][k], [tau("sft", k), tau("hindi", k)]
        if method == "ties":
            m = ties(base, deltas, density, lam=lams[0])
        elif method == "dare":
            m = task_arithmetic(base, [dare(x, p, gen) for x in deltas], lams)
        else:
            m = task_arithmetic(base, deltas, lams)
        out[k] = m.to(dtype)
    return out

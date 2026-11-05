def grads_of(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def global_norm(model):
    # One norm over every gradient in the model, as if they were one long vector
    return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters()))


torch.manual_seed(0)
model = GPT(len(chars), **SMALL)
x, y = Sampler("data/train.bin", block=64, batch=48, seed=0).next_batch()
loss_fn = lambda xb, yb: F.cross_entropy(model(xb).flatten(0, 1), yb.flatten())

model.zero_grad()
loss_fn(x, y).backward()                      # one batch of 48
full = grads_of(model)

N = 4                                          # 4 micro-batches of 12
for divide in [True, False]:
    model.zero_grad()
    for xb, yb in zip(x.chunk(N), y.chunk(N)):
        loss = loss_fn(xb, yb)
        (loss / N if divide else loss).backward()
    acc = grads_of(model)
    print("divide by N" if divide else "no divide  ", "| max diff from full batch", f"{(acc - full).abs().max().item():.1e}",
          "| norm ratio", round(acc.norm().item() / full.norm().item(), 4))

model.zero_grad()
loss_fn(x, y).backward()
norm = global_norm(model)
returned = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print("global norm", round(norm, 4), "| clip_grad_norm_ returned", round(returned.item(), 4),
      "| norm after clipping to 1.0", round(global_norm(model), 4))

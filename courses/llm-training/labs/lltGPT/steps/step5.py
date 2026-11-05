SMALL = dict(vocab=1000, ctx=64, d=64, layers=2, heads=4)

def leak_test(model, T=16, t=7):
    # Change the token at position t + 1 and measure how far each position's logits move
    idx = torch.randint(0, model.cfg.vocab, (1, T))
    changed = idx.clone()
    changed[0, t + 1] = (idx[0, t + 1] + 1) % model.cfg.vocab
    with torch.no_grad():
        moved = (model(idx) - model(changed)).abs().amax(dim=-1)[0]
    return moved[: t + 1].max().item(), moved[t + 1].item()

torch.manual_seed(0)
for name, base in [("gpt2", GPT2), ("modern", MODERN)]:
    model = GPT(replace(base, **SMALL)).eval()
    idx = torch.randint(0, 1000, (2, 16))
    out = model(idx)
    before, at = leak_test(model)
    with torch.no_grad():
        shifted = (model(idx) - model(idx, offset=40)).abs().max().item()
    print(f"{name:6}  logits {tuple(out.shape)}  positions 0-7 moved {before:.1e}  position 8 moved {at:.2f}"
          f"  every position shifted by 40 moved {shifted:.1e}")
    assert out.shape == (2, 16, 1000)
    assert before == 0.0, "a later token changed an earlier output; check the causal mask"
print("shape and leakage tests passed")

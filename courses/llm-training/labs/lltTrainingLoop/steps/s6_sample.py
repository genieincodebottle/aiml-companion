@torch.no_grad()
def generate(path, prompt, n=300, temperature=0.8, seed=0):
    # Rebuild the model from the checkpoint alone: its config, its vocabulary, its weights.
    state = torch.load(path, map_location="cpu", weights_only=False)
    c, vocab = state["cfg"], state["chars"]
    m = GPT(len(vocab), c["block"], c["n_layer"], c["n_head"], c["d"]).to(device).eval()
    m.load_state_dict(state["model"])
    idx = torch.tensor([[vocab.index(ch) for ch in prompt]], device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n):
        logits = m(idx[:, -c["block"]:])[:, -1].float() / temperature
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1, generator=g)], 1)
    return "".join(vocab[i] for i in idx[0].tolist())


print(generate("small-s0.pt", "ROMEO:"))

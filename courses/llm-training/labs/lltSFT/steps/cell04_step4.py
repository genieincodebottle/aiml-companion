model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()

def masked_nll(model, ids, labels):
    x, y = torch.tensor([ids]), torch.tensor(labels)
    with torch.no_grad():
        logits = model(input_ids=x, use_cache=False).logits[0]
    pred, target = logits[:-1], y[1:]           # the logits at position t predict token t + 1
    keep = target != -100
    return torch.nn.functional.cross_entropy(pred[keep], target[keep]).item(), int(keep.sum())

ids = list(tok.apply_chat_template(train[1200]["messages"])["input_ids"])
labels = build_labels(ids)
own, n = masked_nll(model, ids, labels)
with torch.no_grad():
    lib = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]), use_cache=False).loss.item()
every, n_all = masked_nll(model, ids, ids)       # the same example, loss on every token
print("assistant tokens", n, "| own", round(own, 5), "| transformers", round(lib, 5))
print("all", n_all, "tokens | loss", round(every, 5))

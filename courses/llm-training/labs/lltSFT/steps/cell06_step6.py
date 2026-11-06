docs = [list(tok.apply_chat_template(c["messages"])["input_ids"]) for c in train[1200:1203]]
lens = [len(d) for d in docs]
row = torch.tensor([sum(docs, [])])
reset = torch.cat([torch.arange(n) for n in lens])[None]   # positions restart at 0 in each conversation
straight = torch.arange(row.shape[1])[None]                 # naive packing, one long document

def losses(logits):
    out, start = [], 0
    for d in docs:
        y = torch.tensor(build_labels(d))
        keep = y[1:] != -100
        seg = logits[0, start:start + len(d) - 1]
        out.append(round(torch.nn.functional.cross_entropy(seg[keep], y[1:][keep]).item(), 4))
        start += len(d)
    return out

with torch.no_grad():
    print("alone          ", [round(masked_nll(model, d, build_labels(d))[0], 4) for d in docs])
    print("packed, reset  ", losses(model(input_ids=row, position_ids=reset, use_cache=False).logits))
    print("packed, naive  ", losses(model(input_ids=row, position_ids=straight, use_cache=False).logits))
    print("reset, cache on", losses(model(input_ids=row, position_ids=reset, use_cache=True).logits))
print("use_cache in the config after step 5:", model.config.use_cache)

doc_of = torch.tensor([0, 0, 0, 1, 1])                       # two documents of 3 and 2 tokens
causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
print((causal & (doc_of[:, None] == doc_of[None, :])).int())

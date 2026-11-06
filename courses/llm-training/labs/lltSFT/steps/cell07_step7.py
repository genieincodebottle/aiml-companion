torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

def pack(convs, max_len=1024):
    rows, cur = [], []
    for c in convs:
        ids = list(tok.apply_chat_template(c["messages"])["input_ids"])
        if cur and sum(len(d) for d in cur) + len(ids) > max_len:
            rows.append(cur)
            cur = []
        cur.append(ids)
    return rows + [cur]

def heldout_loss(model, convs):
    total = count = 0
    for c in convs:
        ids = list(tok.apply_chat_template(c["messages"])["input_ids"])
        loss, n = masked_nll(model, ids, build_labels(ids))
        total += loss * n
        count += n
    return total / count

small = train[1200:1224] + train[2400:2424]                  # 48 short rewrite and summarize examples
check = held[100:124] + held[200:224]                         # 48 held-out ones from the same splits
rows = pack(small)
model.eval()
print(len(rows), "packed rows | held-out loss before", round(heldout_loss(model, check), 4))
model.train()
for step, docs in enumerate(rows):
    x = torch.tensor([sum(docs, [])])
    pos = torch.tensor([sum((list(range(len(d))) for d in docs), [])])
    y = torch.tensor(sum((build_labels(d) for d in docs), []))
    logits = model(input_ids=x, position_ids=pos, use_cache=False).logits[0]   # use_cache=False keeps the document mask
    keep = y[1:] != -100
    loss = torch.nn.functional.cross_entropy(logits[:-1][keep], y[1:][keep])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    if step % 8 == 0:
        print("step", step, "train loss", round(loss.item(), 4))
model.eval()
print("held-out loss after", round(heldout_loss(model, check), 4))

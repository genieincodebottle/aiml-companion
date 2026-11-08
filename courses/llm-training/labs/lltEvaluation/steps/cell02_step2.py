def encode_pair(tok, context, continuation):
    # As lm-eval does: move the context's trailing spaces onto the continuation, encode the joined
    # text once, and call every token after the context's tokens the continuation.
    n = len(context) - len(context.rstrip())
    if n:
        context, continuation = context[:-n], context[-n:] + continuation
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]

@torch.no_grad()
def logliks(model, tok, pairs, batch_size=32):
    # Summed log-probability of each continuation given its context. Longest first, right-padded.
    enc = [encode_pair(tok, c, k) for c, k in pairs]
    order = sorted(range(len(enc)), key=lambda i: -(len(enc[i][0]) + len(enc[i][1])))
    body, head = model.get_decoder(), model.get_output_embeddings()
    out = np.zeros(len(enc))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [enc[i][0] + enc[i][1] for i in idx]
        ids = torch.zeros((len(idx), max(len(x) for x in seqs)), dtype=torch.long)
        att = torch.zeros_like(ids)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = torch.tensor(x)
            att[r, :len(x)] = 1
        hidden = body(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
        for r, i in enumerate(idx):
            c, k = enc[i]
            pos = torch.arange(len(c) - 1, len(c) + len(k) - 1, device=device)   # the positions that predict k
            logp = torch.log_softmax(head(hidden[r, pos]).float(), dim=-1)          # float32 softmax
            out[i] = float(logp[torch.arange(len(k)), torch.tensor(k, device=device)].sum())
    return out

def score_items(lls_per_item, choice_lens, golds):
    # acc: the choice with the highest summed log-probability. acc_norm: divide by the choice's length in characters.
    acc = [int(np.argmax(l) == g) for l, g in zip(lls_per_item, golds)]
    acc_norm = [int(np.argmax(np.array(l) / np.array(n)) == g) for l, n, g in zip(lls_per_item, choice_lens, golds)]
    return np.array(acc), np.array(acc_norm)

# Rescore the harness's own ARC-Easy requests with your code, on the same loaded model.
lm = LMS[BASE]
reqs = HARNESS[BASE]["arcRequests"]
pairs = [p for r in reqs for p in r["pairs"]]
t = time.time()
ours = logliks(lm.model, lm.tokenizer, pairs)
theirs = np.array([x for r in reqs for x in r["lls"]])
cuts = np.cumsum([0] + [len(r["pairs"]) for r in reqs])
ours_items = [ours[cuts[j]:cuts[j + 1]] for j in range(len(reqs))]
lens = [[len(k) - 1 for _, k in r["pairs"]] for r in reqs]            # the choice text without its leading space
acc, acc_norm = score_items(ours_items, lens, [r["target"] for r in reqs])
gap = np.abs(ours - theirs)
print(len(pairs), "continuations rescored in", round(time.time() - t, 1), "s")
print("log-likelihood gap to the harness | max", round(float(gap.max()), 4), "| mean", round(float(gap.mean()), 5))
print("acc", round(acc.mean(), 3), "vs harness", round(np.mean(HARNESS[BASE]["tasks"]["arc_easy"]["acc"]), 3),
      "| items scored differently:", int((acc != np.array(HARNESS[BASE]["tasks"]["arc_easy"]["acc"])).sum()))
print("acc_norm", round(acc_norm.mean(), 3), "vs harness", round(np.mean(HARNESS[BASE]["tasks"]["arc_easy"]["acc_norm"]), 3))
pick_differs = sum(int(np.argmax(l) != np.argmax(np.array(l) / np.array(n))) for l, n in zip(ours_items, lens))
print("items where acc and acc_norm pick different choices:", pick_differs)
json.dump({"n": len(pairs), "maxGap": float(gap.max()), "meanGap": float(gap.mean()), "seconds": round(time.time() - t, 1),
           "acc": acc.tolist(), "accNorm": acc_norm.tolist(), "pickDiffers": pick_differs,
           "itemsDiffer": int((acc != np.array(HARNESS[BASE]["tasks"]["arc_easy"]["acc"])).sum())}, open("results/scorer.json", "w"))

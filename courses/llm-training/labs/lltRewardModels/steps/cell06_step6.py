def score(model, seqs, budget=16384):
    # one reward per token-id list; batches of similar length, at most 32 rows or 16,384 tokens each
    out, order, k = [None] * len(seqs), sorted(range(len(seqs)), key=lambda i: -len(seqs[i])), 0
    while k < len(order):
        idx = order[k:k + max(1, min(32, budget // len(seqs[order[k]])))]
        ids, mask = pad([seqs[i] for i in idx])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            r = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1).float().tolist()
        for i, v in zip(idx, r):
            out[i] = v
        k += len(idx)
    return out

def held_out_report(model, pairs):
    s = score(model, [p["c"] for p in pairs] + [p["r"] for p in pairs])
    rc, rr = np.array(s[:len(pairs)]), np.array(s[len(pairs):])
    win = rc > rr
    longer = np.array([len(p["c"]) > len(p["r"]) for p in pairs])
    shorter = np.array([len(p["c"]) < len(p["r"]) for p in pairs])
    tie = np.array([p["sc"] == p["sr"] for p in pairs])
    boot = np.random.default_rng(0).integers(0, len(win), (10000, len(win)))
    lo, hi = np.percentile(win[boot].mean(1), [2.5, 97.5])
    lengths = np.array([len(p["c"]) for p in pairs] + [len(p["r"]) for p in pairs])
    print("accuracy", round(win.mean(), 3), "95% CI", round(lo, 3), "to", round(hi, 3),
          "| without equal-score pairs", round(win[~tie].mean(), 3))
    print("chosen longer", round(win[longer].mean(), 3), "of", longer.sum(), "| chosen shorter", round(win[shorter].mean(), 3),
          "of", shorter.sum(), "| length-only rule", round(longer.mean(), 3))
    print("correlation of reward with length", round(np.corrcoef(np.concatenate([rc, rr]), lengths)[0, 1], 3))
    return win

win_s0 = held_out_report(rm, held)

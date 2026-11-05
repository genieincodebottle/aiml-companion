from collections import Counter
from prompts import PROMPTS          # 40 prompts in five groups of 8, written by the notebook's first cell

rows, responses = [], {}
for p in PROMPTS:
    ctx = chat(p)
    resp = greedy(inst, tok_i, ctx, 128)
    responses[p] = resp
    kl, rank = kl_and_rank(logprobs(inst, ctx, resp), logprobs(base, ctx, resp), resp)
    for t, r in enumerate(resp):
        if r != tok_i.eos_token_id:                  # the end token is scored on its own below
            rows.append((t + 1, float(kl[t]), int(rank[t]), tok_i.decode([r]).strip() or repr(tok_i.decode([r]))))

pos = np.array([r[0] for r in rows]); kl_all = np.array([r[1] for r in rows]); rk = np.array([r[2] for r in rows])
print(len(rows), "response tokens from", len(PROMPTS), "prompts")
print("unshifted, base rank 1  ", round(float((rk == 1).mean()), 3))
print("marginal, base rank 2-3 ", round(float(((rk > 1) & (rk <= 3)).mean()), 3))
print("shifted, base rank > 3  ", round(float((rk > 3).mean()), 3))
for lo, hi in [(1, 1), (2, 5), (6, 20), (21, 60), (61, 128)]:
    m = (pos >= lo) & (pos <= hi)
    print("positions", lo, "to", hi, "|", round(float(m.mean()), 3), "of tokens |",
          round(float(kl_all[m].sum() / kl_all.sum()), 3), "of the KL")
print(Counter(r[3] for r in rows if r[2] > 3).most_common(12))

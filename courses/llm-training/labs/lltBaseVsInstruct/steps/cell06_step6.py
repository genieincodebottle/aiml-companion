PLAIN = """Question: {p}
Answer:
"""

rows_plain = []
for p in PROMPTS:
    resp = responses[p]
    ctx_i, ctx_b = chat(p), tok_b(PLAIN.format(p=p))["input_ids"]
    kl, rank = kl_and_rank(logprobs(inst, ctx_i, resp), logprobs(base, ctx_b, resp), resp)
    for t, r in enumerate(resp):
        if r != tok_i.eos_token_id:
            rows_plain.append((t + 1, float(kl[t]), int(rank[t])))

pos_p = np.array([r[0] for r in rows_plain]); kl_p = np.array([r[1] for r in rows_plain]); rk_p = np.array([r[2] for r in rows_plain])
for name, k, r, ps in [("chat tokens", kl_all, rk, pos), ("plain question", kl_p, rk_p, pos_p)]:
    print(name.ljust(15), "mean KL", round(float(k.mean()), 3), "| unshifted", round(float((r == 1).mean()), 3),
          "| shifted", round(float((r > 3).mean()), 3), "| KL in first 5 tokens", round(float(k[ps <= 5].sum() / k.sum()), 3))

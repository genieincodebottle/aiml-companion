from transformers import AutoModelForSequenceClassification
GOLD = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"         # the gold judge, also the labeller in step 11
JUDGE2 = "Skywork/Skywork-Reward-V2-Llama-3.2-1B"      # a second judge on another base model and tokenizer

def score_seqs(model, seqs, pad_id, budget=12288):
    # one reward per token-id list: batches of similar length, at most 32 rows or 12,288 tokens, right padding
    order, out, k = sorted(range(len(seqs)), key=lambda i: -len(seqs[i])), [None] * len(seqs), 0
    while k < len(order):
        idx = order[k:k + max(1, min(32, budget // len(seqs[order[k]])))]
        ids = torch.full((len(idx), len(seqs[idx[0]])), pad_id)
        att = torch.zeros_like(ids)
        for i, j in enumerate(idx):
            ids[i, :len(seqs[j])], att[i, :len(seqs[j])] = torch.tensor(seqs[j]), 1
        with torch.no_grad():
            r = model(input_ids=ids.to(GPU), attention_mask=att.to(GPU)).logits.squeeze(-1).float().tolist()
        for i, j in enumerate(idx):
            out[j] = r[i]
        k += len(idx)
    return np.array(out)

def gold_scores(sets, judge=GOLD):
    # sets: {name: list of (prompt, reply)}. Loads the judge once, scores in its own chat format, frees it.
    gtok = AutoTokenizer.from_pretrained(judge)
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gold_rm = AutoModelForSequenceClassification.from_pretrained(judge, num_labels=1, dtype=torch.float16).to(GPU).eval()
    gold_rm.config.pad_token_id = gtok.pad_token_id
    out = {}
    for name, items in sets.items():
        seqs = []
        for q, a in items:
            s = gtok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=False)
            seqs.append(gtok(s, add_special_tokens=False).input_ids[:4096])     # the template already holds any BOS
        out[name] = score_seqs(gold_rm, seqs, gtok.pad_token_id)
    del gold_rm
    gc.collect()
    torch.cuda.empty_cache()
    return out

def paired(a, b, seed=0):
    # win rate of a over b on the same prompts (a tie counts half) and the mean gold difference, 95% bootstrap intervals
    w = 0.5 + 0.5 * np.sign(a - b)
    boot = np.random.default_rng(seed).integers(0, len(w), (10000, len(w)))
    wl, wh = np.percentile(w[boot].mean(1), [2.5, 97.5])
    dl, dh = np.percentile((a - b)[boot].mean(1), [2.5, 97.5])
    return {"win": round(float(w.mean()), 3), "winCI": [round(float(wl), 3), round(float(wh), 3)],
            "diff": round(float((a - b).mean()), 3), "diffCI": [round(float(dl), 3), round(float(dh), 3)]}

gens = {"dpo-b0.1-s0": generate(dpo, eval_prompts)}
with dpo.disable_adapter():                       # adapter off = the SFT model, same seeds
    gens["sft"] = generate(dpo, eval_prompts)
del dpo
gc.collect()
torch.cuda.empty_cache()
gold = gold_scores({k: list(zip(eval_prompts, [g["text"] for g in v])) for k, v in gens.items()})
judge2 = gold_scores({k: list(zip(eval_prompts, [g["text"] for g in v])) for k, v in gens.items()}, JUDGE2)
for k in ["sft", "dpo-b0.1-s0"]:
    print(k, "| gold mean", round(float(gold[k].mean()), 3), "| reply tokens", round(float(np.mean([g["tokens"] for g in gens[k]])), 1),
          "| ended", round(float(np.mean([g["ended"] for g in gens[k]])), 3))
runs["judge"], runs["judge2"] = paired(gold["dpo-b0.1-s0"], gold["sft"]), paired(judge2["dpo-b0.1-s0"], judge2["sft"])
print("DPO against SFT on", len(eval_prompts), "prompts, gold judge:", runs["judge"])
print("second judge:", runs["judge2"])

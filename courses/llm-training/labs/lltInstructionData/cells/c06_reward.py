import gzip
from transformers import AutoModelForSequenceClassification

RM = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm_tok = AutoTokenizer.from_pretrained(RM)
rm_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
rm = AutoModelForSequenceClassification.from_pretrained(RM, dtype=rm_dtype, num_labels=1).to(device).eval()

@torch.no_grad()
def reward(pairs, batch_size=8, max_len=2048):
    # One scalar per (prompt, response). Shortest first, so each batch pads little.
    ids = [rm_tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": a}],
                                      tokenize=True)["input_ids"][:max_len] for p, a in pairs]
    order = np.argsort([len(x) for x in ids])
    out = np.zeros(len(ids), dtype=np.float32)
    for s in range(0, len(ids), batch_size):
        idx = order[s:s + batch_size]
        batch = rm_tok.pad({"input_ids": [ids[k] for k in idx]}, return_tensors="pt").to(device)
        out[idx] = rm(**batch).logits[:, 0].float().cpu().numpy()
    return out

# Calibrate on human labels first: HelpSteer2 validation, single-turn rows, helpfulness rated 0 to 4.
path = hf_hub_download("nvidia/HelpSteer2", "validation.jsonl.gz", repo_type="dataset",
                       revision="990b2711a36180dd19d9c94b8627844866f8982a")
hs = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
hs = [h for h in hs if "<extra_id_1>" not in h["prompt"]]
hs_score = reward([(h["prompt"], h["response"]) for h in hs])
helpful = np.array([h["helpfulness"] for h in hs])
by_prompt = collections.defaultdict(list)
for k, h in enumerate(hs):
    by_prompt[h["prompt"]].append(k)
pairs = [(a, b) for ks in by_prompt.values() for a in ks for b in ks if a < b and helpful[a] != helpful[b]]
agree = np.mean([(hs_score[a] > hs_score[b]) == (helpful[a] > helpful[b]) for a, b in pairs])
print(len(hs), "single-turn HelpSteer2 responses |", len(pairs), "pairs with different human helpfulness")
print("reward model agrees with the human order on", round(float(agree), 3), "of pairs")
for q in [0.25, 0.5, 0.75]:
    t = float(np.quantile(hs_score, q))
    print("keep score >=", round(t, 2), "| kept", int((hs_score >= t).sum()),
          "| share rated 3 or 4:", round(float((helpful[hs_score >= t] >= 3).mean()), 3),
          "| dropped share rated 3 or 4:", round(float((helpful[hs_score < t] >= 3).mean()), 3))

RM_FLOOR = float(np.quantile(hs_score, 0.5))
scores = reward([(r["prompt"], r["response"]) for r in pool])
for src in ["dolci", "wildchat", "synthetic"]:
    m = np.array([r["source"] == src for r in pool])
    if m.any():
        print(src.ljust(9), "rows", int(m.sum()), "| median reward", round(float(np.median(scores[m])), 2))
del rm

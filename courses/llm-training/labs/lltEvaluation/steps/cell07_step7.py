import random
from transformers import AutoModelForCausalLM, AutoTokenizer

if device == "cuda":
    print("GPU memory in use before the canary:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

# Leak 100 ARC-Easy test items into a small training set on purpose, then see what the leak does and what finds it.
CANARY = "HuggingFaceTB/SmolLM2-360M"
arc_all = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", revision="210d026faf9955653af8916fad021475a3f00453")
n_leak = 8 if SMOKE else 100
members = list(arc_all.select(range(1000, 1000 + n_leak)))          # leaked into training
others = list(arc_all.select(range(1000 + n_leak, 1000 + 2 * n_leak)))   # never seen, same distribution

def item_text(doc):                                  # the item exactly as the harness formats it, with its answer
    return "Question: " + doc["question"] + NL + "Answer: " + doc["choices"]["text"][gold_of(doc)]

dolly = load_dataset("databricks/databricks-dolly-15k", split="train", revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
rows = dolly.shuffle(seed=0).select(range(64 if SMOKE else 1600))
docs = ["Instruction: " + r["instruction"] + NL + ("Context: " + r["context"] + NL if r["context"] else "") + "Response: " + r["response"] for r in rows]
docs += [item_text(d) for d in members] * 3          # each leaked item appears three times
random.Random(0).shuffle(docs)

c_tok = AutoTokenizer.from_pretrained(CANARY)
c_tok.pad_token = c_tok.eos_token

@torch.no_grad()
def probe(model):
    # Cloze accuracy, Min-K% score and verbatim completion for leaked and unseen items.
    out = {}
    for group, items in [("leaked", members), ("unseen", others)]:
        pairs, spans = [], []
        for d in items:
            spans.append((len(pairs), len(pairs) + len(d["choices"]["text"])))
            pairs += [("Question: " + d["question"] + NL + "Answer:", " " + c) for c in d["choices"]["text"]]
        ll = logliks(model, c_tok, pairs)
        acc = [int(np.argmax(ll[a:b]) == gold_of(d)) for (a, b), d in zip(spans, items)]
        mink = []
        for d in items:                              # Min-K% Prob: mean log-prob of the 20% least likely tokens
            ids = torch.tensor([c_tok(item_text(d))["input_ids"]], device=device)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1).gather(-1, ids[0, 1:, None])[:, 0]
            k = max(1, int(0.2 * len(lp)))
            mink.append(float(lp.sort().values[:k].mean()))
        heads, tails = [], []
        for d in items:                              # give the first half of the question, check the next 5 words
            w = d["question"].split()
            h = len(w) // 2
            heads.append("Question: " + " ".join(w[:h]))
            tails.append(" ".join(w[h:h + 5]))
        gens = generate(model, c_tok, heads, max_new=16, batch_size=64)
        verbatim = [int(" ".join(g.split()).startswith(t)) for (g, _), t in zip(gens, tails)]
        out[group] = {"acc": acc, "minK": mink, "verbatim": verbatim}
    return out

def auroc(pos, neg):
    # Probability that a random leaked item scores higher than a random unseen one (0.5 = no signal).
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())

def report(tag, r):
    print(tag.ljust(7), "| cloze acc leaked", round(np.mean(r["leaked"]["acc"]), 3), "unseen", round(np.mean(r["unseen"]["acc"]), 3),
          "| verbatim leaked", round(np.mean(r["leaked"]["verbatim"]), 3), "unseen", round(np.mean(r["unseen"]["verbatim"]), 3),
          "| Min-K% AUROC", round(auroc(r["leaked"]["minK"], r["unseen"]["minK"]), 3))

canary = AutoModelForCausalLM.from_pretrained(CANARY, dtype=torch.float32).to(device)   # fp32 weights, fp16 compute
CANARY_RES = {"before": probe(canary.eval())}
report("before", CANARY_RES["before"])

STEPS_C, BS, MAXLEN = (4 if SMOKE else 200), 16, 256
enc = [c_tok(d, truncation=True, max_length=MAXLEN)["input_ids"] + [c_tok.eos_token_id] for d in docs]
opt = torch.optim.AdamW(canary.parameters(), lr=3e-5, weight_decay=0.0)
scaler = torch.amp.GradScaler(device, init_scale=2048, enabled=device == "cuda")
rng = np.random.default_rng(0)
canary.gradient_checkpointing_enable()               # recompute activations, so the 1.7B model can stay loaded
canary.train()
t = time.time()
for step in range(STEPS_C):
    batch = [enc[i][:MAXLEN] for i in rng.choice(len(enc), size=BS, replace=False)]
    ids = torch.full((BS, max(len(x) for x in batch)), c_tok.pad_token_id)
    att = torch.zeros_like(ids)
    for r, x in enumerate(batch):
        ids[r, :len(x)], att[r, :len(x)] = torch.tensor(x), 1
    labels = ids.masked_fill(att == 0, -100)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        loss = canary(input_ids=ids.to(device), attention_mask=att.to(device), labels=labels.to(device)).loss
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    if step % 50 == 0 or step == STEPS_C - 1:
        print("step", step, "loss", round(loss.item(), 3), "|", round(time.time() - t), "s")
CANARY_RES["after"] = probe(canary.eval())
report("after", CANARY_RES["after"])

# The defender's check when you hold the training data: word 8-gram overlap with every ARC-Easy test item.
def grams(s, n=8):
    w = "".join(ch if ch.isalnum() else " " for ch in s.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
train_grams = set().union(*[grams(d) for d in docs])
flagged = [j for j, d in enumerate(arc_all) if (g := grams(item_text(d))) and len(g & train_grams) / len(g) >= 0.5]
leaked_ids = set(range(1000, 1000 + n_leak))
print("8-gram check | flagged", len(flagged), "of", len(arc_all), "test items | leaked items caught", len(leaked_ids & set(flagged)), "of", n_leak,
      "| flagged but never leaked", len(set(flagged) - leaked_ids))
CANARY_RES["ngram"] = {"flagged": flagged, "nTest": len(arc_all)}
json.dump(CANARY_RES, open("results/canary.json", "w"))
del canary, opt
gc.collect()
torch.cuda.empty_cache()

import gc, math
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TEXT, N_REPLY, N_GEN = (200, 300, 100) if DEV.type == "cuda" else (20, 30, 0)   # a CPU scores a small slice, no generation
tok = AutoTokenizer.from_pretrained(BASE)
NL = chr(10)
TEMPLATE = "### Instruction:" + NL + "{}" + NL + NL + "### Response:" + NL     # the plain-text template of the lltDPO SFT model
runs = {}                                                                      # every number this notebook prints is kept here

def cut(text, n_chars):
    # The first n_chars characters, cut back to the last space (the lltDomainAdaptation panel recipe)
    return text[:n_chars].rsplit(" ", 1)[0] if len(text) > n_chars else None

def held_texts(name, config, split, n_docs, n_chars, want):
    out = []
    for r in load_dataset(name, name=config, split=split, streaming=True).take(n_docs):
        t = cut(r["text"], n_chars)
        if t and len(tok(t).input_ids) <= 1023:
            out.append(t)
        if len(out) == want:
            break
    return out

hindi = held_texts("HuggingFaceFW/fineweb-2", "hin_Deva", "test", 400, 1000, N_TEXT)
english = held_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 500, 3000, N_TEXT)
replies, prompts = [], []
for ex in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs"):
    p = tok(TEMPLATE.format(ex["prompt"])).input_ids
    c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
    full = tok(TEMPLATE.format(ex["prompt"]) + c).input_ids + [tok.eos_token_id]
    if c != r and len(full) <= 512 and len(tok(TEMPLATE.format(ex["prompt"]) + r).input_ids) + 1 <= 512:
        replies.append((full, len(p)))
        if len(p) <= 200:
            prompts.append(ex["prompt"])
    if len(replies) == N_REPLY:
        break
prompts = prompts[:N_GEN]
print(len(hindi), "Hindi texts |", len(english), "English texts |", len(replies), "held-out replies |", len(prompts), "prompts")

@torch.no_grad()
def nll_items(model, rows, starts, bs):
    # Summed NLL in nats and token count of each row, scoring the tokens from position starts[i] on
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    nll, cnt = np.zeros(len(rows)), np.zeros(len(rows))
    for j in range(0, len(order), bs):
        idx = order[j:j + bs]
        L = max(len(rows[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.eos_token_id)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for a, i in enumerate(idx):
            ids[a, :len(rows[i])] = torch.tensor(rows[i])
            att[a, :len(rows[i])] = 1
        logits = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None].to(DEV))[..., 0]
        for a, i in enumerate(idx):
            nll[i] = -float(lp[a, starts[i] - 1:len(rows[i]) - 1].sum())
            cnt[i] = len(rows[i]) - starts[i]
    return nll, cnt

@torch.no_grad()
def end_rate(model, max_new=256, bs=25):
    # Greedy replies to held-out prompts: does the model stop on its own with <|endoftext|>?
    tok.padding_side = "left"
    ended, lengths, texts = [], [], []
    for j in range(0, len(prompts), bs):
        enc = tok([TEMPLATE.format(p) for p in prompts[j:j + bs]], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for row in out[:, enc.input_ids.shape[1]:].tolist():
            stop = tok.eos_token_id in row
            n = row.index(tok.eos_token_id) + 1 if stop else len(row)
            ended.append(float(stop)); lengths.append(n); texts.append(tok.decode(row[:n], skip_special_tokens=True))
    return np.array(ended), np.array(lengths), texts

H_ROWS = [[tok.eos_token_id] + tok(t).input_ids for t in hindi]
E_ROWS = [[tok.eos_token_id] + tok(t).input_ids for t in english]
H_BYTES = np.array([len(t.encode("utf-8")) for t in hindi])
E_BYTES = np.array([len(t.encode("utf-8")) for t in english])

def panel(model):
    # The two skills and one forgetting check, with per-item numbers kept for paired intervals
    h, _ = nll_items(model, H_ROWS, [1] * len(H_ROWS), bs=2)
    e, _ = nll_items(model, E_ROWS, [1] * len(E_ROWS), bs=2)
    r, n = nll_items(model, [x[0] for x in replies], [x[1] for x in replies], bs=4)
    out = {"hiBpb": h.sum() / math.log(2) / H_BYTES.sum(), "enBpb": e.sum() / math.log(2) / E_BYTES.sum(),
           "replyNll": r.sum() / n.sum(), "items": {"hi": h, "en": e, "reply": r, "replyTok": n}}
    if prompts:
        ended, lengths, texts = end_rate(model)
        out.update(endRate=ended.mean(), meanTokens=lengths.mean(), items=dict(out["items"], ended=ended), sample=texts[0])
    return out

def ci(a, b, key, n=10000, seed=0):
    # 95% paired bootstrap interval of (b minus a) over the same items
    num = {"hi": ("hi", H_BYTES), "en": ("en", E_BYTES), "reply": ("reply", "replyTok"), "ended": ("ended", None)}[key]
    rng = np.random.default_rng(seed)
    N = len(a["items"][num[0]])
    i = rng.integers(0, N, (n, N))
    def stat(m):
        top = m["items"][num[0]][i].sum(1)
        if num[1] is None:
            return top / N
        den = num[1][i].sum(1) * math.log(2) if isinstance(num[1], np.ndarray) else m["items"][num[1]][i].sum(1)
        return top / den
    d = stat(b) - stat(a)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16 if DEV.type == "cuda" else torch.float32).to(DEV).eval()

def load_into(sd):
    # Put a state dict into the one model on the GPU; lm_head is tied to the embeddings, so it is not in the file
    missing, unexpected = model.load_state_dict({k: v.to(model.dtype) for k, v in sd.items()}, strict=False)
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)

def show(name, m, ref=None):
    line = name.ljust(22) + " | Hindi bpb " + format(m["hiBpb"], ".4f") + " | English bpb " + format(m["enBpb"], ".4f")
    line += " | reply NLL " + format(m["replyNll"], ".4f")
    if "endRate" in m:
        line += " | ends " + format(m["endRate"], ".2f") + ", " + format(m["meanTokens"], ".0f") + " tokens"
    print(line)

t0 = time.time()
for name in ("base", "sft", "hindi"):
    load_into(W[name])
    runs[name] = panel(model)
    show(name, runs[name])
print("Hindi bpb, hindi minus base", ci(runs["base"], runs["hindi"], "hi"), "| reply NLL, sft minus base",
      ci(runs["base"], runs["sft"], "reply"), "|", round(time.time() - t0), "s")

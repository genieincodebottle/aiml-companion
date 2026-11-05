"""Exactness measurements for lltTinyStories that run on a laptop CPU (T1).

python measure.py tokleak      tokenizer trained on validation text: bytes per token, seen vs unseen halves
python measure.py bpb          bits per byte two ways on our step-3,000 checkpoint (byte table vs decode)
python measure.py prompts      continuation loss for prompts copied from training stories vs from val stories
python measure.py resumecheck  weights-only reload of our step-2,000 checkpoint reproduces its val loss

Every figure goes to ../../runs/lltTinyStories/lltTinyStories-T1-<name>.json.
Timing figures are NOT recorded here (shared machine); see measure_timing.py.
"""
import datetime, json, math, os, platform, sys
import numpy as np
import torch, torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(ROOT, "runs", "lltTinyStories")
CKPT_DIR = os.environ.get("LLT_CKPT_DIR", os.path.join(ROOT, "kaggle", "out", "llt-tinystories-pretrain-test", "ckpt"))


def env():
    import tokenizers
    import datasets
    return {"python": platform.python_version(), "torch": torch.__version__, "tokenizers": tokenizers.__version__,
            "numpy": np.__version__, "datasets": datasets.__version__, "cpu": "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM", "torchThreads": 4}


def write(name, label, result, config=None):
    rid = "lltTinyStories-T1-" + name + os.environ.get("LLT_RUN_SUFFIX", "")
    rec = {"runId": rid, "module": "lltTinyStories", "tier": "T1", "label": label,
           "hardware": {"gpu": None, "count": 0}, "env": env(), "config": config or {}, "seed": 0,
           "evals": result, "date": datetime.date.today().isoformat(), "command": "python measure.py " + " ".join(sys.argv[1:])}
    path = os.path.join(RUNS, rid + ".json")
    json.dump(rec, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=1)); print("wrote", path)


def data():
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    return ds["train"], ds["validation"]["text"]


def new_tok(texts, vocab=4096):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<|eot|>"],
                                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def our_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(CKPT_DIR, "tokenizer.json"))


def load_model(step, path=None):
    sys.path.insert(0, HERE)
    from model_def import GPT
    ck = torch.load(path or os.path.join(CKPT_DIR, "step%d.pt" % step), map_location="cpu")
    c = ck["cfg"]
    m = GPT(c["vocab"], c["d"], c["layers"], c["heads"], c["ctx"])
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return m.eval(), c


def encode_stories(tok, texts):
    eot = tok.token_to_id("<|eot|>")
    out = []
    for e in tok.encode_batch(texts):
        out.extend(e.ids); out.append(eot)
    return np.array(out, dtype=np.int64)


def token_bytes(tok):
    eot = tok.token_to_id("<|eot|>")
    return np.array([1 if i == eot else len(tok.id_to_token(i)) for i in range(tok.get_vocab_size())], dtype=np.int64)


@torch.no_grad()
def window_nll(model, ids, starts, T):
    """Summed NLL and token count over windows ids[s:s+T] predicting ids[s+1:s+T+1]."""
    nll, n = 0.0, 0
    for i in range(0, len(starts), 16):
        ss = starts[i:i + 16]
        x = torch.tensor(np.stack([ids[s:s + T] for s in ss]))
        y = torch.tensor(np.stack([ids[s + 1:s + T + 1] for s in ss]))
        lg = model(x)
        nll += F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1), reduction="sum").item(); n += y.numel()
    return nll, n


def cmd_tokleak():
    train, val = data()
    half = len(val) // 2
    va, vb, tail = val[:half], val[half:], train["text"][-half:]
    def bpt(tok, texts):
        return sum(len(s.encode()) for s in texts) / sum(len(e.ids) for e in tok.encode_batch(texts))
    toks = {"ours (300k train stories)": our_tok(), "leaky (val half A)": new_tok(va), "control (train tail, same size)": new_tok(tail)}
    res = {name: {k: round(bpt(t, x), 4) for k, x in [("valA", va), ("valB", vb), ("trainTail", tail)]} for name, t in toks.items()}
    for name in res:
        res[name]["gapA_minus_B"] = round(res[name]["valA"] - res[name]["valB"], 4)
    res["note"] = "valA = first 10,995 validation stories (the leaky tokenizer trained on them); valB = the other 10,995; trainTail = last 10,995 training stories"
    write("tokleak", "broken:tokenizer-trained-on-val", res, {"vocab": 4096})


def cmd_bpb():
    tok = our_tok(); _, val = data()
    ids = encode_stories(tok, val); eot = tok.token_to_id("<|eot|>")
    tb = token_bytes(tok)
    model, c = load_model(3000)
    T = c["ctx"]; starts = list(range(0, 640 * T, T))
    nll, n = window_nll(model, ids, starts, T)
    ys = np.stack([ids[s + 1:s + T + 1] for s in starts])
    bytes_table = int(tb[ys].sum())
    bytes_decode = sum(len(tok.decode([int(t) for t in row if t != eot]).encode()) + int((row == eot).sum()) for row in ys)
    text_bytes = sum(len(s.encode()) for s in val) + len(val)
    res = {"evalTokens": n, "meanNll": round(nll / n, 4), "bitsPerToken": round(nll / n / math.log(2), 4),
           "bytesByTable": bytes_table, "bytesByDecode": bytes_decode,
           "bpbTable": round(nll / (math.log(2) * bytes_table), 4), "bpbDecode": round(nll / (math.log(2) * bytes_decode), 4),
           "bytesPerTokenWindow": round(bytes_table / n, 4),
           "wholeValCheck": {"tableBytes": int(tb[ids].sum()), "utf8BytesPlusOnePerStory": text_bytes},
           "kaggleReported": {"valLoss": 1.6445, "valBpb": 0.5934, "dtype": "fp16 autocast on T4"}}
    write("bpb-check", "check:bits-per-byte", res, {"checkpoint": "lltTinyStories-T2-baseline-s0 step 3000", "evalWindows": 640, "ctx": T})


def cmd_prompts(n=300, p_len=48, c_len=64, seed=0, ckpt=None):
    tok = our_tok(); train, val = data()
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train), 20 * n, replace=False)
    tr_text = [train[int(i)]["text"] for i in tr_idx]
    out = {}
    models_to_test = [("baseline-s0 step 3000", None)]
    if ckpt:
        models_to_test.append((os.path.basename(ckpt), ckpt))
    for name, path in models_to_test:
        model, c = load_model(3000, path)
        rows = {}
        for split, texts in [("trainPrompts", tr_text), ("valPrompts", list(val))]:
            seqs = [e.ids for e in tok.encode_batch(texts) if len(e.ids) >= p_len + c_len]
            pick = rng.permutation(len(seqs))[:n] if split == "valPrompts" else range(min(n, len(seqs)))
            per = []
            with torch.no_grad():
                for i in pick:
                    s = np.array(seqs[i][:p_len + c_len], dtype=np.int64)
                    lg = model(torch.tensor(s[None, :-1]))[0, p_len - 1:]
                    per.append(F.cross_entropy(lg, torch.tensor(s[p_len:])).item())
            rows[split] = np.array(per)
        d = rows["trainPrompts"].mean() - rows["valPrompts"].mean()
        boots = [rng.choice(rows["trainPrompts"], n).mean() - rng.choice(rows["valPrompts"], n).mean() for _ in range(2000)]
        out[name] = {"trainPromptLoss": round(rows["trainPrompts"].mean(), 4), "valPromptLoss": round(rows["valPrompts"].mean(), 4),
                     "difference": round(d, 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                     "n": n}
    write("prompts-copied", "broken:prompts-from-training-stories", out, {"promptTokens": p_len, "continuationTokens": c_len})


def cmd_resumecheck():
    tok = our_tok(); _, val = data()
    ids = encode_stories(tok, val)
    model, c = load_model(2000)
    T = c["ctx"]; nll, n = window_nll(model, ids, list(range(0, 640 * T, T)), T)
    write("resume-check", "check:weights-only-reload", {"step2000ValLossCpuFp32": round(nll / n, 4), "kaggleReportedStep2000": 1.7323},
          {"checkpoint": "lltTinyStories-T2-baseline-s0 step 2000"})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompts":
        cmd_prompts(ckpt=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"tokleak": cmd_tokleak, "bpb": cmd_bpb, "resumecheck": cmd_resumecheck}[cmd]()

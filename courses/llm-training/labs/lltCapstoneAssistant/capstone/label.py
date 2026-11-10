"""Stage 3b. Score every sampled reply with a reward model and build preference pairs.

chosen   = the highest-scoring reply that ended on its own (did not hit the token cap)
rejected = the lowest-scoring reply
A prompt gives a pair only if the score gap is at least --min-gap, so near-ties do not become noisy labels.
The reward model reads each (prompt, reply) in its own chat template, as its model card says.
Also scores everything with a second, smaller reward model when --second is given, to measure how often
the T2 labeller would have picked the same pairs.
"""
import math

from .common import RM, TEMPLATE, Timer, cli, done, path, read_jsonl, stage_record, write_json, write_jsonl


def score(rm_name, convs, cfg, batch=16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rm_name)
    kw = {"dtype": getattr(torch, cfg["dtype"]), "num_labels": 1}
    if cfg["quant"] == "nf4" and "8B" in rm_name:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=getattr(torch, cfg["dtype"]))
    rm = AutoModelForSequenceClassification.from_pretrained(rm_name, device_map={"": 0} if torch.cuda.is_available() else None, **kw).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm.config.pad_token_id = tok.pad_token_id        # the score is read at the last non-pad token
    texts = [tok.apply_chat_template(c, tokenize=False) for c in convs]
    if tok.bos_token and texts and texts[0].startswith(tok.bos_token):
        texts = [t[len(tok.bos_token):] for t in texts]                # the tokenizer adds BOS itself
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=3072).to(rm.device)
        with torch.no_grad():
            s = rm(**enc).logits[:, 0].float().tolist()
        for i, v in zip(idx, s):
            out[i] = v
        if (k // batch) % 50 == 0:
            print("scored", k + len(idx), "/", len(order), flush=True)
    del rm
    torch.cuda.empty_cache()
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return round(sxy / (sx * sy), 4) if sx and sy else None


def build_pairs(rows, key, min_gap):
    pairs, skipped = [], {"no reply ended on its own": 0, "gap below min": 0}
    for r in rows:
        reps = r["replies"]
        ended = [o for o in reps if o["finished"]]
        if not ended:
            skipped["no reply ended on its own"] += 1
            continue
        best = max(ended, key=lambda o: o[key])
        worst = min(reps, key=lambda o: o[key])
        if best[key] - worst[key] < min_gap or best is worst:
            skipped["gap below min"] += 1
            continue
        pairs.append({"id": r["id"], "prompt": TEMPLATE.format(r["prompt"]), "raw_prompt": r["prompt"],
                      "chosen": best["text"], "rejected": worst["text"], "score_chosen": best[key], "score_rejected": worst[key],
                      "len_chosen": best["tokens"], "len_rejected": worst["tokens"]})
    return pairs, skipped


def main():
    a = cli("Stage 3b: reward-model labels and pairs", [(("--min-gap",), dict(type=float, default=1.0)),
                                                         (("--second",), dict(default=None))])
    cfg = a.cfg
    out = path(a, "pairs", "pairs.jsonl")
    if done(out):
        print("stage 3b already done")
        return
    rows = read_jsonl(path(a, "pairs", "samples.jsonl"))
    flat = [(i, j) for i, r in enumerate(rows) for j in range(len(r["replies"]))]
    convs = [[{"role": "user", "content": rows[i]["prompt"]}, {"role": "assistant", "content": rows[i]["replies"][j]["text"]}]
             for i, j in flat]
    with Timer() as t:
        s1 = score(RM[a.tier], convs, cfg)
    for (i, j), v in zip(flat, s1):
        rows[i]["replies"][j]["rm"] = v
    extra = {}
    if a.second:
        with Timer() as t2:
            s2 = score(a.second, convs, cfg)
        for (i, j), v in zip(flat, s2):
            rows[i]["replies"][j]["rm2"] = v
        p2, _ = build_pairs(rows, "rm2", a.min_gap)
        extra["second"] = {"model": a.second, "sec": t2.sec, "pearsonWithMain": pearson(s1, s2), "pairs": len(p2)}
    write_jsonl(path(a, "pairs", "scored.jsonl"), rows)
    pairs, skipped = build_pairs(rows, "rm", a.min_gap)
    if a.second:
        # Same prompt in both pair sets: does the second model pick the same chosen and rejected replies?
        p2 = {p["id"]: p for p in build_pairs(rows, "rm2", a.min_gap)[0]}
        both = [p for p in pairs if p["id"] in p2]
        extra["second"]["samePairShare"] = round(sum(p["chosen"] == p2[p["id"]]["chosen"] and p["rejected"] == p2[p["id"]]["rejected"]
                                                     for p in both) / max(1, len(both)), 4)
        extra["second"]["sameChosenShare"] = round(sum(p["chosen"] == p2[p["id"]]["chosen"] for p in both) / max(1, len(both)), 4)
        extra["second"]["sharedPrompts"] = len(both)
    write_jsonl(out, pairs)
    lens = [o["tokens"] for r in rows for o in r["replies"]]
    sc = [o["rm"] for r in rows for o in r["replies"]]
    stage_record(a, "label", {
        "wallClockSec": t.sec, "rm": RM[a.tier], "minGap": a.min_gap, "n_scored": len(sc), "n_pairs": len(pairs), "skipped": skipped,
        "scoreLengthPearson": pearson(sc, lens),
        "chosenLongerShare": round(sum(p["len_chosen"] > p["len_rejected"] for p in pairs) / max(1, len(pairs)), 4),
        "meanLenChosen": round(sum(p["len_chosen"] for p in pairs) / max(1, len(pairs)), 1),
        "meanLenRejected": round(sum(p["len_rejected"] for p in pairs) / max(1, len(pairs)), 1),
        "meanGap": round(sum(p["score_chosen"] - p["score_rejected"] for p in pairs) / max(1, len(pairs)), 3),
        **extra,
    })


if __name__ == "__main__":
    main()

from datasets import load_dataset

LETTERS = "ABCDE"
arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(LIMIT))   # the harness's items

def arc_formats(doc):
    # Five ways to ask the same question: (context, continuations, choice texts for acc_norm).
    q, ch = doc["question"], doc["choices"]["text"]
    let = [LETTERS[i] for i in range(len(ch))]
    lettered = "".join(NL + let[i] + ". " + c for i, c in enumerate(ch))
    paren = "".join(NL + "(" + let[i].lower() + ") " + c for i, c in enumerate(ch))
    return {
        "Question/Answer cloze": ("Question: " + q + NL + "Answer:", [" " + c for c in ch], ch),   # lm-eval's arc_easy
        "Q/A cloze": ("Q: " + q + NL + "A:", [" " + c for c in ch], ch),
        "bare cloze": (q, [" " + c for c in ch], ch),
        "lettered, MMLU style": ("Question: " + q + lettered + NL + "Answer:", [" " + x for x in let], let),
        "lettered, (a) style": (q + paren + NL + "The answer is (", [x.lower() for x in let], let),
    }

def gold_of(doc):
    return doc["choices"]["label"].index(doc["answerKey"])

def score_formats(lm):
    # One batched pass over all five formats of every item; returns per-item acc for each format.
    items = [arc_formats(doc) for doc in arc]
    names = list(items[0])
    pairs, spans = [], []
    for it in items:
        for f in names:
            ctx, conts, _ = it[f]
            spans.append((len(pairs), len(pairs) + len(conts)))
            pairs += [(ctx, k) for k in conts]
    ll = logliks(lm.model, lm.tokenizer, pairs)
    golds = [gold_of(doc) for doc in arc]
    res, k = {f: [] for f in names}, 0
    for j, it in enumerate(items):
        for f in names:
            a, b = spans[k]
            res[f].append(int(np.argmax(ll[a:b]) == golds[j]))
            k += 1
    return {f: np.array(v) for f, v in res.items()}

FORMATS = {name: score_formats(LMS[name]) for name in [BASE, TUNED]}
print("format".ljust(22), "base".rjust(6), "tuned".rjust(6), "gain".rjust(7))
for f in FORMATS[BASE]:
    b, t = FORMATS[BASE][f].mean(), FORMATS[TUNED][f].mean()
    print(f.ljust(22), str(round(b, 3)).rjust(6), str(round(t, 3)).rjust(6), str(round(t - b, 3)).rjust(7))
spread = {name: max(v.mean() for v in FORMATS[name].values()) - min(v.mean() for v in FORMATS[name].values()) for name in FORMATS}
print("best minus worst format | base", round(spread[BASE], 3), "| tuned", round(spread[TUNED], 3))
json.dump({n: {f: v.tolist() for f, v in d.items()} for n, d in FORMATS.items()}, open("results/formats.json", "w"))

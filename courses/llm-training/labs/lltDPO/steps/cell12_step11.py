RUN_EXPERIMENT = False   # True samples 2,000 replies from the SFT model, labels them twice and trains three DPO models

def proxy_scores(sets):
    # our 360M reward model from lltRewardModels, in its own chat template, read at the last real token
    path = os.path.join(snapshot_download("aimlcompanion/llm-training-lab", allow_patterns="lltRewardModels/rm-s0/*"), "lltRewardModels", "rm-s0")
    rtok = AutoTokenizer.from_pretrained(path)
    rm = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16).to(GPU).eval()
    out = {name: score_seqs(rm, [list(rtok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}])["input_ids"])[:1024]
                                 for q, a in items], rtok.pad_token_id) for name, items in sets.items()}
    del rm
    gc.collect()
    torch.cuda.empty_cache()
    return out

def make_pair(q, x, y, sx, sy):
    # the higher-scored reply becomes chosen; None unless both ended, differ, are scored differently and fit MAXLEN
    if not (x["ended"] and y["ended"]) or x["text"] == y["text"] or sx == sy:
        return None
    win, lose = (x, y) if sx > sy else (y, x)
    p, ci = encode(q, win["text"])
    ri = encode(q, lose["text"])[1]
    if len(ci) > MAXLEN or len(ri) > MAXLEN:
        return None
    return {"prompt": q, "chosen": win["text"], "rejected": lose["text"], "p": p, "c": ci, "r": ri}

if RUN_EXPERIMENT:
    prompts = [x["prompt"] for x in dpo_pairs]
    pol, t0 = load_sft(), time.time()
    a, b = generate(pol, prompts, temperature=1.0, seed=0, batch=64), generate(pol, prompts, temperature=1.0, seed=1, batch=64)
    runs["onpolicySampleSec"] = round(time.time() - t0, 1)
    on = {"a": list(zip(prompts, [x["text"] for x in a])), "b": list(zip(prompts, [x["text"] for x in b]))}
    off = {"a": [(x["prompt"], x["chosen"]) for x in dpo_pairs], "b": [(x["prompt"], x["rejected"]) for x in dpo_pairs]}
    g = gold_scores({"on_a": on["a"], "on_b": on["b"], "off_a": off["a"], "off_b": off["b"]})
    pr = proxy_scores(on)
    # keep a prompt only when both labellers give it an on-policy pair, so all three arms use the same prompts and size
    keep, sets = [], {"onpolicy-gold-s0": [], "onpolicy-proxy-s0": [], "offline-gold-s0": []}
    for i, q in enumerate(prompts):
        pg = make_pair(q, a[i], b[i], g["on_a"][i], g["on_b"][i])
        pp = make_pair(q, a[i], b[i], pr["a"][i], pr["b"][i])
        if pg is None or pp is None or g["off_a"][i] == g["off_b"][i]:
            continue
        keep.append(i)
        sets["onpolicy-gold-s0"].append(pg)
        sets["onpolicy-proxy-s0"].append(pp)
        x = dpo_pairs[i]                       # the same prompt's UltraFeedback replies, relabelled by the gold model
        sets["offline-gold-s0"].append(dict(x) if g["off_a"][i] > g["off_b"][i] else
                                       dict(x, chosen=x["rejected"], rejected=x["chosen"], c=x["r"], r=x["c"]))
    longer = lambda ps: round(float(np.mean([len(x["c"]) > len(x["r"]) for x in ps])), 3)
    runs["onpolicy-labels"] = {"prompts": len(prompts), "kept": len(keep), "ended": round(float(np.mean([x["ended"] for x in a + b])), 3),
                               "goldAgreesProxy": round(float(np.mean([(g["on_a"][i] > g["on_b"][i]) == (pr["a"][i] > pr["b"][i]) for i in keep])), 3),
                               "goldKeepsUltraFeedbackLabel": round(float(np.mean([g["off_a"][i] > g["off_b"][i] for i in keep])), 3),
                               "chosenLonger": {k: longer(v) for k, v in sets.items()}}
    print(runs["onpolicy-labels"], "| sampling took", runs["onpolicySampleSec"], "s")
    epochs = max(1, round(len(dpo_pairs) / max(1, len(keep))))       # about as many steps as the 1,000-pair runs
    for name, ps in sets.items():
        # held-out checks only at the start and the end of each arm
        experiment([(name, "dpo", 0.1, 0, {"every": 10 ** 6, "epochs": epochs})], ps, reference_logps(pol, ps))
    del pol
    gc.collect()
    torch.cuda.empty_cache()

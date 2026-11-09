def pack(prompt_ids, comp_ids):
    # right-padded rows of prompt + completion, an attention mask, and a mask over the completion tokens
    L = max(len(p) + len(c) for p, c in zip(prompt_ids, comp_ids))
    ids = torch.full((len(prompt_ids), L), tok.pad_token_id)
    att, cm = torch.zeros_like(ids), torch.zeros_like(ids)
    for i, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        att[i, :len(p) + len(c)] = 1
        cm[i, len(p):len(p) + len(c)] = 1
    return ids.to(GPU), att.to(GPU), cm.to(GPU)

def logits_at(model, ids, att, cm):
    # fp32 logits only at the positions that predict a completion token: [tokens, vocab], never [rows, length, vocab]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast("cuda", dtype=torch.float16):
        h = base.model(input_ids=ids, attention_mask=att).last_hidden_state
        return base.lm_head(h[:, :-1][cm[:, 1:].bool()]).float()

def kl_report(model, prompt_ids, comp_ids, micro=8):
    # mean per-token forward KL, reverse KL and student entropy over the completion tokens, in nats
    tot = dict(forwardKL=0.0, reverseKL=0.0, studentEntropy=0.0)
    n = 0
    for i in range(0, len(prompt_ids), micro):
        ids, att, cm = pack(prompt_ids[i:i + micro], comp_ids[i:i + micro])
        with torch.no_grad():
            lps = logits_at(model, ids, att, cm).log_softmax(-1)
            lpt = logits_at(teacher, ids, att, cm).log_softmax(-1)
            tot["forwardKL"] += float((lpt.exp() * (lpt - lps)).sum())
            tot["reverseKL"] += float((lps.exp() * (lps - lpt)).sum())
            tot["studentEntropy"] += float(-(lps.exp() * lps).sum())
        n += int(cm[:, 1:].sum())
        del lps, lpt
    return {k: round(v / n, 4) for k, v in tot.items()}

R = {}

def evaluate(model, name, batch=100):
    # greedy answers to the test questions, accuracy, and KL to the teacher on the model's own answers and on the teacher's
    t0 = time.time()
    comp = generate(model, [prompt(x["q"]) for x in test], batch=batch)
    gen_sec = time.time() - t0
    texts = [tok.decode(c, skip_special_tokens=True) for c in comp]
    correct = [extract(t) is not None and extract(t) == gold(x["a"]) for t, x in zip(texts, test)]
    rec = dict(acc=round(float(np.mean(correct)), 4), correct=correct, stopped=round(float(np.mean([cut(t) != t for t in texts])), 3),
               meanTokens=round(float(np.mean([len(c) for c in comp])), 1), genSec=round(gen_sec, 1), texts=texts)
    if model is not teacher:
        rec["onOwnAnswers"] = kl_report(model, test_prompt_ids, comp)
        rec["onTeacherAnswers"] = kl_report(model, test_prompt_ids, teacher_test)
    R[name] = rec
    json.dump(R, open("distill_results.json", "w"))
    print(name, "| accuracy", rec["acc"], "| stopped", rec["stopped"], "| mean answer tokens", rec["meanTokens"],
          "| generation s", rec["genSec"], "| own answers", rec.get("onOwnAnswers"), "| teacher answers", rec.get("onTeacherAnswers"))
    return comp

teacher_test = evaluate(teacher, "teacher", batch=64)
print(R["teacher"]["texts"][0][:600])
evaluate(student, "student, before training")
print(R["student, before training"]["texts"][0][:600])
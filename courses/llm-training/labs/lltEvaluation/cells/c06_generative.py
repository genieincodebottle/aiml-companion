from lm_eval.tasks.ifeval.utils import process_results

def chat(tok, messages):
    # The chat prompt, ending where the assistant starts. Qwen3 thinking is switched off; other templates ignore the flag.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def generate(model, tok, texts, max_new, batch_size=32):
    # Greedy continuations of ready-made prompt strings, left-padded. Returns (text, ended) pairs.
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    outs = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           eos_token_id=tok.eos_token_id, pad_token_id=pad)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append((tok.decode(row, skip_special_tokens=True), bool((row == tok.eos_token_id).any())))
    return outs

def first_letter(text, n):
    # The first standalone capital A to E in the reply, or -1. "A." and "(B)" count; the A in "Apple" does not.
    t = " " + text + " "
    for i in range(1, len(t) - 1):
        if t[i] in LETTERS[:n] and not t[i - 1].isalnum() and not t[i + 1].isalnum():
            return LETTERS.index(t[i])
    return -1

def letter_prompt(doc):
    ch = doc["choices"]["text"]
    return "Question: " + doc["question"] + "".join(NL + LETTERS[i] + ". " + c for i, c in enumerate(ch))

# Generative scoring of ARC-Easy: the model writes its answer and a parser reads the letter.
N_GEN = min(32 if SMOKE else 500, len(arc))
GEN = {}
for name in [BASE, TUNED]:
    lm = LMS[name]
    docs = list(arc.select(range(N_GEN)))
    if name == TUNED:
        texts = [chat(lm.tokenizer, [{"role": "user", "content": letter_prompt(d) + NL + "Answer with the letter only."}]) for d in docs]
    else:
        texts = [letter_prompt(d) + NL + "Answer:" for d in docs]
    t = time.time()
    outs = generate(lm.model, lm.tokenizer, texts, max_new=8, batch_size=64)
    parsed = [first_letter(o, len(d["choices"]["text"])) for (o, _), d in zip(outs, docs)]
    correct = np.array([int(p == gold_of(d)) for p, d in zip(parsed, docs)])
    GEN[name] = {"correct": correct.tolist(), "parsed": [int(p >= 0) for p in parsed], "replies": [o for o, _ in outs[:5]]}
    print(name, "| generated letters: acc", round(correct.mean(), 3), "| replies with a letter", round(np.mean(GEN[name]["parsed"]), 3),
          "| log-likelihood letters on the same items", round(FORMATS[name]["lettered, MMLU style"][:N_GEN].mean(), 3), "|", round(time.time() - t, 1), "s")
    print("  first replies:", [o[:30] for o, _ in outs[:3]])

del LMS[BASE]                                        # the base model is done; free its memory for long generations
gc.collect()
torch.cuda.empty_cache()

# IFEval: 541 prompts with checkable instructions ("answer in under 100 words", "no commas"), scored by lm-eval's own checker.
ifeval = load_dataset("google/IFEval", split="train", revision="966cd89545d6b6acfd7638bc708b98261ca58e84")
N_IF = 16 if SMOKE else 300                          # the first 300 of the 541 prompts

def ifeval_scores(model, tok, max_new=512):
    docs = list(ifeval.select(range(N_IF)))
    outs = generate(model, tok, [chat(tok, [{"role": "user", "content": d["prompt"]}]) for d in docs], max_new=max_new, batch_size=64)
    res = [process_results(d, [text]) for d, (text, _) in zip(docs, outs)]
    return {"strict": [int(r["prompt_level_strict_acc"]) for r in res], "loose": [int(r["prompt_level_loose_acc"]) for r in res],
            "ended": [int(e) for _, e in outs]}

t = time.time()
IFEVAL = {"original": ifeval_scores(LMS[TUNED].model, LMS[TUNED].tokenizer)}
s = IFEVAL["original"]
print(TUNED, "| IFEval prompt-level strict", round(np.mean(s["strict"]), 3), "| loose", round(np.mean(s["loose"]), 3),
      "| replies that ended", round(np.mean(s["ended"]), 3), "|", len(s["strict"]), "prompts in", round(time.time() - t), "s")
json.dump({"letters": GEN, "ifeval": IFEVAL}, open("results/generative.json", "w"))

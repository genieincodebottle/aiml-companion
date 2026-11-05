"""Steps 2 to 5 and 7 on a CPU: our scorer, five prompt formats, generative scoring, MMLU.

Writes per-item results to <work>/scoring_<model>.json. measure.py turns them into Run Library files.
Usage: python measure_scoring.py <work dir> [base|instruct]
"""
import json, os, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer import logliks, score_item, arc_formats, arc_gold, parse_letter, LETTERS

torch.set_num_threads(4)
torch.manual_seed(0)
work, which = sys.argv[1], sys.argv[2]
NAME = {"base": "HuggingFaceTB/SmolLM2-360M", "instruct": "HuggingFaceTB/SmolLM2-360M-Instruct"}[which]
N_ARC, N_MMLU_PER_SUBJECT = int(os.environ.get("LLT_N_ARC", 500)), int(os.environ.get("LLT_N_MMLU", 20))

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(N_ARC))
R = {"model": NAME, "nArc": N_ARC, "formats": {}, "generative": {}, "mmlu": {}}
t0 = time.time()

# Five prompt formats, log-likelihood scoring, same 500 items.
for i, doc in enumerate(arc):
    gold = arc_gold(doc)
    for f, (ctx, conts, texts) in arc_formats(doc).items():
        lls = logliks(model, tok, ctx, conts)
        acc, acc_norm = score_item(lls, texts, gold)
        d = R["formats"].setdefault(f, {"acc": [], "acc_norm": [], "lls": []})
        d["acc"].append(acc); d["acc_norm"].append(acc_norm); d["lls"].append([round(x, 5) for x in lls])
    if i % 100 == 0:
        print("arc", i, round(time.time() - t0), flush=True)

# Generative scoring: the model writes a letter, we parse the first A-E.
@torch.no_grad()
def generate(prompt_ids):
    out = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                         max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)

modes = ["raw"] + (["chat_template"] if tok.chat_template else [])
for mode in modes:
    res = {"correct": [], "parsed": [], "text": []}
    for doc in arc:
        ch = doc["choices"]["text"]
        lettered = "".join(f"\n{LETTERS[i]}. {c}" for i, c in enumerate(ch))
        user = f"Question: {doc['question']}{lettered}\nAnswer with the letter of the correct option."
        if mode == "chat_template":
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)["input_ids"]
        else:
            ids = tok(user + "\nAnswer:", return_tensors="pt")["input_ids"]
        text = generate(ids)
        p = parse_letter(text, len(ch))
        res["correct"].append(int(p == arc_gold(doc))); res["parsed"].append(int(p >= 0)); res["text"].append(text)
    R["generative"][mode] = res
    print("gen", mode, np.mean(res["correct"]), np.mean(res["parsed"]), round(time.time() - t0), flush=True)

# MMLU: the first 20 test items of each of the 57 subjects, lettered and cloze.
mm = load_dataset("cais/mmlu", "all", split="test")
by_subject = {}
for j, s in enumerate(mm["subject"]):
    if len(by_subject.setdefault(s, [])) < N_MMLU_PER_SUBJECT:
        by_subject[s].append(j)
idx = sorted(j for v in by_subject.values() for j in v)
for f in ["letters", "cloze"]:
    R["mmlu"][f] = {"acc": [], "subject": []}
for j in idx:
    doc = mm[j]
    head = f"The following are multiple choice questions (with answers) about {doc['subject'].replace('_', ' ')}.\n\n"
    q, ch, gold = doc["question"].strip(), doc["choices"], doc["answer"]
    lettered = "".join(f"\n{LETTERS[i]}. {c}" for i, c in enumerate(ch))
    lls = logliks(model, tok, f"{head}{q}{lettered}\nAnswer:", [" " + x for x in LETTERS[:4]])
    R["mmlu"]["letters"]["acc"].append(int(np.argmax(lls) == gold))
    lls = logliks(model, tok, f"{head}Question: {q}\nAnswer:", [" " + c for c in ch])
    R["mmlu"]["cloze"]["acc"].append(int(np.argmax(lls) == gold))
    for f in ["letters", "cloze"]:
        R["mmlu"][f]["subject"].append(doc["subject"])
R["mmlu"]["nSubjects"] = len(by_subject); R["mmlu"]["n"] = len(idx)
R["wallClockSec"] = round(time.time() - t0, 1)
print("mmlu", {f: np.mean(R["mmlu"][f]["acc"]) for f in ["letters", "cloze"]}, flush=True)
with open(os.path.join(work, f"scoring_{which}.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh)
print("done", R["wallClockSec"])

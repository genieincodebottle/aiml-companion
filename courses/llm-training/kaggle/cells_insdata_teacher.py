"""Kaggle kernel llt-insdata-teacher: pick the Magpie teacher for lltInstructionData by measurement.

Three Apache-2.0 candidates on one T4, fp16, the page's Magpie recipe (pre-query template, temperature 1.0,
then a greedy answer with thinking off). Per teacher: tokens/s, the share of usable instructions, the share
of answers that end inside 512 tokens, the reward of the answers (Skywork-Reward-V2-Qwen3-0.6B) and the mean
cosine similarity of the instructions (granite-embedding-311m-multilingual-r2). Writes insdata_teacher.json.
"""
import os, runpy
head = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labs", "lltInstructionData", "notebook_head.py"))

CELLS = [
("md", "# lltInstructionData teacher bake-off on a T4"),
("code", head["RUNTIME_CHECK"]),
("code", head["INSTALL"]),
("code", r'''
import os, gc, json, time, platform, collections
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0
R = {"gpu": torch.cuda.get_device_name(0), "env": {"python": platform.python_version(), "torch": torch.__version__,
     "cuda": torch.version.cuda, "transformers": transformers.__version__}, "teachers": {}}
NL = chr(10)
N, BS = 128, 64

def gen(model, tok, prompts, max_new, sample, BS=64):
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    outs, ntok, t0 = [], 0, time.time()
    for s in range(0, len(prompts), BS):
        enc = tok(prompts[s:s + BS], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None,
                               top_p=1.0 if sample else None, top_k=0 if sample else None,
                               eos_token_id=end_id, pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for row in new:
            hit = (row == end_id).nonzero()
            n = int(hit[0, 0]) + 1 if len(hit) else len(row)
            ntok += n
            outs.append((tok.decode(row[:n], skip_special_tokens=True).strip(), bool(len(hit))))
    torch.cuda.synchronize()
    return outs, ntok, time.time() - t0

def usable(q):
    if not (15 <= len(q) <= 1500) or "<think>" in q or q.lower().startswith(("assistant", "user", "system")):
        return False
    try:
        return detect(q[:500]) == "en"
    except Exception:
        return False

for name in ["Qwen/Qwen3.5-2B", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B"]:
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    load = time.time() - t
    torch.manual_seed(0)
    ins, n1, s1 = gen(model, tok, ["<|im_start|>user" + NL] * N, 128, True)
    qs = [q for q, _ in ins]
    chats = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for q in qs]
    ans, n2, s2 = gen(model, tok, chats, 512, False, BS=32 if "4B" in name else 64)
    R["teachers"][name] = {"class": type(model).__name__, "loadSec": round(load, 1),
        "instrTokPerSec": round(n1 / s1, 1), "answerTokPerSec": round(n2 / s2, 1), "instrSec": round(s1, 1), "answerSec": round(s2, 1),
        "usableShare": float(np.mean([usable(q) for q in qs])), "answersEnded": float(np.mean([e for _, e in ans])),
        "peakMemGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "instructions": qs, "answers": [a for a, _ in ans]}
    print(name, {k: v for k, v in R["teachers"][name].items() if k not in ("instructions", "answers")}, flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    json.dump(R, open("insdata_teacher.json", "w"), indent=1)
'''),
("code", r'''
rm_tok = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
rm = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B", dtype=torch.float16, num_labels=1).to("cuda").eval()
emb = SentenceTransformer("ibm-granite/granite-embedding-311m-multilingual-r2", device="cuda", model_kwargs={"dtype": torch.float32})
emb.max_seq_length = 512
for name, T in R["teachers"].items():
    sc = []
    for q, a in zip(T["instructions"], T["answers"]):
        ids = rm_tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                         tokenize=True, return_tensors="pt", return_dict=True).to("cuda")
        ids = {k: v[:, :2048] for k, v in ids.items()}
        with torch.no_grad():
            sc.append(float(rm(**ids).logits[0, 0].float()))
    X = emb.encode(T["instructions"], normalize_embeddings=True, convert_to_numpy=True)
    n = len(X)
    T["rewards"] = sc
    T["medianReward"] = float(np.median(sc))
    T["meanCosine"] = float(((X @ X.T).sum() - n) / (n * (n - 1)))
    lam = np.linalg.eigvalsh(X @ X.T / n); lam = lam[lam > 1e-12]
    T["vendi"] = float(np.exp(-(lam * np.log(lam)).sum()))
    print(name, "median reward", round(T["medianReward"], 3), "| mean cosine", round(T["meanCosine"], 3), "| Vendi", round(T["vendi"], 1))
    for q in T["instructions"][:6]:
        print("   -", q[:140].replace(NL, " "))
json.dump(R, open("insdata_teacher.json", "w"), indent=1)
'''),
]

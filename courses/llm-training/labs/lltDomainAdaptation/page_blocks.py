"""The code blocks of the lltDomainAdaptation page, in page order. make_notebook.py builds the notebook from them
and export_blocks.py writes them into the page draft, so the page shows exactly the code the Kaggle run executed.
Rules for page code: no backslash, no backtick, no dollar-brace (the page stores code in JS template literals)."""

PANEL = '''import gc, json, time, torch
import dalab as L
from transformers import AutoModelForCausalLM, AutoTokenizer

hi_test = L.stream_docs("HuggingFaceFW/fineweb-2", "test", 1000, "hin_Deva")
en_edu = L.stream_docs("HuggingFaceFW/fineweb-edu", "train", 8000, "sample-10BT")
hi_train = L.stream_docs("HuggingFaceFW/fineweb-2", "train", 20000, "hin_Deva")
tok = AutoTokenizer.from_pretrained(L.MODEL)
E = {"hi": L.eval_texts(hi_test[:400], 1000, 200, tok),      # 200 held-out Hindi texts, 1,000 characters each
     "en": L.eval_texts(en_edu[:500], 3000, 200, tok),       # 200 held-out English texts, 3,000 characters each
     "xsc_hi": L.load_xstorycloze("hi"), "xsc_en": L.load_xstorycloze("en"), "hellaswag": L.load_hellaswag(2000)}
print({k: len(v) for k, v in E.items()})

model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32, device_map="cuda:0").eval()
t = time.time()
base_panel = L.panel(model, tok, E)
print(base_panel, round(time.time() - t), "s")'''

FERT = '''texts = [d["text"] for d in hi_test]
words = sum(len(x.split()) for x in texts)
for mid in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-0.8B-Base", "HuggingFaceTB/SmolLM2-360M"]:
    t = AutoTokenizer.from_pretrained(mid)
    n = sum(len(ids) for ids in t(texts, add_special_tokens=False)["input_ids"])
    pieces = sum(len(t.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)) for x in texts)
    print(mid.ljust(28), len(t), "in vocab |", round(n / words, 2), "tokens per word |", round(pieces / words, 2), "pieces per word")

s = "भारत की राजधानी"
for mid in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-0.8B-Base"]:
    pre = AutoTokenizer.from_pretrained(mid).backend_tokenizer.pre_tokenizer
    print(mid, [s[a:b] for _, (a, b) in pre.pre_tokenize_str(s)])'''

RAG = '''from sentence_transformers import SentenceTransformer
import numpy as np

SPLIT = json.load(open("qa_split.json", encoding="utf-8"))  # written by the notebook: which questions are test and train
contexts, qa = L.load_xquad_hi()                             # 240 Hindi paragraphs, 1,190 questions
byid = {x["id"]: x for x in qa}
pool = contexts + [t for t in (L.cut(d["text"], 1000) for d in hi_test[200:1000]) if t]   # plus web distractors
emb = SentenceTransformer("intfloat/multilingual-e5-small")
P = emb.encode(pool, prompt="passage: ", normalize_embeddings=True, batch_size=64)
test = [byid[i] for i in SPLIT["testIds"]]
Qe = emb.encode([x["q"] for x in test], prompt="query: ", normalize_embeddings=True, batch_size=64)
top = np.argsort(-(Qe @ P.T), axis=1)[:, :5]
del emb; gc.collect(); torch.cuda.empty_cache()                # free the embedder before training
for k in (1, 2, 5):
    print("recall@" + str(k), round(float(np.mean([x["ctx"] in top[i, :k] for i, x in enumerate(test)])), 4))
print(len(pool), "paragraphs in the pool,", len(test), "test questions")'''

QA = '''Q = {"test": test, "train": [byid[i] for i in SPLIT["trainIds"]], "shots": SPLIT["shots"], "pool": pool,
     "contexts": contexts, "retrieved": {x["id"]: [int(j) for j in top[i]] for i, x in enumerate(test)}}
print(L.qa_prompt(test[0]["q"], [pool[j] for j in top[0, :2]])[-300:])
t = time.time()
base_qa = L.qa_eval(model, tok, Q)                           # closed0, closed3, rag0, rag3
print({m: (r["em"], r["f1"]) for m, r in base_qa.items()}, round(time.time() - t), "s")'''

CPT = '''import random

STEPS, REPLAY, PEAK_LR = QUICK_STEPS_VALUE, 0.05, 3e-5
docs = [d["text"] for d in hi_train[:4000]]
rng = random.Random(1234)
for c in contexts * 4:                                       # the 240 paragraphs, 4 copies each, among the web docs
    docs.insert(rng.randrange(0, 1000), c)
n_rows = STEPS * 8                                           # 8 rows of 1,024 tokens per optimizer step
hi_rows = L.pack_rows(docs, tok, 1024, n_rows - round(n_rows * REPLAY))
en_rows = L.pack_rows([d["text"] for d in en_edu[500:]], tok, 1024, max(1, round(n_rows * REPLAY)))
rows, n_en = L.mix_rows(hi_rows, en_rows, n_rows, REPLAY, seed=0)
print(rows.shape, n_en, "English rows")

gc.collect(); torch.cuda.empty_cache()                       # return cached blocks left by step 3 generation
print("GPU before training:", round(torch.cuda.memory_allocated() / 2**30, 2), "GiB allocated,",
      round(torch.cuda.memory_reserved() / 2**30, 2), "GiB reserved")
base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
quick = lambda m: {"hiBpb": round(L.bits_per_byte(m, tok, E["hi"][:64])["bpb"], 4),
                   "enBpb": round(L.bits_per_byte(m, tok, E["en"][:64])["bpb"], 4)}
cfg = dict(steps=STEPS, rows_per_step=8, micro=4, peak_lr=PEAK_LR, eval_every=50)
log, peak, sec = L.train_cpt(model, rows, cfg, quick)
print("peak GB", peak, "| train s", sec, "| tokens/s", round(n_rows * 1024 / sec))
cpt_panel = L.panel(model, tok, E)
print(cpt_panel)'''

INTERP = '''ft_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
for a in (0.25, 0.5, 0.75):
    L.interpolate(model, base_sd, ft_sd, a)                  # theta(a) = (1 - a) theta_base + a theta_cpt
    print(a, L.panel(model, tok, E, with_tasks=False))
L.interpolate(model, base_sd, ft_sd, 1.0)'''

EXT = '''from extend_tokenizer import extend

tok_dir, rep = extend(L.MODEL, [d["text"] for d in hi_train[:20000]], 10000, False, "qwen3-hi-ext")
ext = AutoTokenizer.from_pretrained(tok_dir)
for name, t in (("Qwen3", tok), ("extended", ext)):
    n = sum(len(ids) for ids in t(texts, add_special_tokens=False)["input_ids"])
    print(name.ljust(9), len(t), "in vocab |", round(n / words, 2), "tokens per Hindi word")
print(rep)
print("endoftext", ext.convert_tokens_to_ids("<|endoftext|>"), "| im_end", ext.convert_tokens_to_ids("<|im_end|>"))'''

INIT = '''parts = json.load(open(tok_dir + "/new_token_parts.json"))     # new id -> the Qwen3 ids it is made of
first = sorted(parts, key=int)[:3]
print({i: ext.convert_ids_to_tokens(int(i)) for i in first}, {i: parts[i] for i in first})

def add_tokens_mean_init(model, n_vocab, parts):
    model.resize_token_embeddings(n_vocab, pad_to_multiple_of=64, mean_resizing=False)
    E_in = model.get_input_embeddings().weight               # tied: the output layer shares this matrix
    with torch.no_grad():
        for new_id, old_ids in parts.items():
            E_in[int(new_id)] = E_in[old_ids].mean(0)
    return model

ext_model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32, device_map="cuda:0").eval()
add_tokens_mean_init(ext_model, len(ext), parts)
print(ext_model.get_input_embeddings().weight.shape, ext_model.get_output_embeddings().weight is ext_model.get_input_embeddings().weight)
print("Hindi bpb, step 0", round(L.bits_per_byte(ext_model, ext, E["hi"][:100])["bpb"], 4),
      "| English bpb, step 0", round(L.bits_per_byte(ext_model, ext, E["en"][:100])["bpb"], 4))
del ext_model; gc.collect(); torch.cuda.empty_cache()'''

QLORA = '''from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

del model; gc.collect(); torch.cuda.empty_cache()          # the fp32 model of steps 1 to 7 leaves the GPU
print("GPU before QLoRA:", round(torch.cuda.memory_allocated() / 2**30, 2), "GiB allocated")
q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16)     # fp16 compute: the T4 has no bf16 tensor cores
qm = AutoModelForCausalLM.from_pretrained(L.MODEL, quantization_config=q4, dtype=torch.float16, device_map="cuda:0")
for p in qm.parameters():
    p.requires_grad = False
qm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
qm.enable_input_require_grads()
qm = get_peft_model(qm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
for p in qm.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()                              # fp32 adapters, so the fp16 GradScaler can unscale them
print(sum(p.numel() for p in qm.parameters() if p.requires_grad), "trainable parameters")
ex = L.qa_train_examples(tok, Q, seed=0)                     # gold paragraph + one other, loss on the answer only
print(len(ex), "examples |", tok.decode([t for t, lab in zip(*ex[0]) if lab != -100]))
log, peak, sec = L.train_lora_qa(qm, tok, ex, epochs=2, bs=8, micro=2, lr=2e-4)
print(len(log), "steps | loss", log[0]["trainLoss"], "->", log[-1]["trainLoss"], "| peak GB", peak, "| s", sec)'''

QLORA_EVAL = '''t = time.time()
ft_qa = L.qa_eval(qm, tok, Q, modes=("closed0", "rag0"))
print({m: (r["em"], r["f1"]) for m, r in ft_qa.items()}, round(time.time() - t), "s")
for e in ft_qa["rag0"]["examples"][:3]:
    print(e)'''

ORDER = [("PANEL", PANEL), ("FERT", FERT), ("RAG", RAG), ("QA", QA), ("CPT", CPT), ("INTERP", INTERP),
         ("EXT", EXT), ("INIT", INIT), ("QLORA", QLORA), ("QLORA_EVAL", QLORA_EVAL)]
for name, code in ORDER:
    assert chr(92) not in code and chr(96) not in code and "$" + "{" not in code, name

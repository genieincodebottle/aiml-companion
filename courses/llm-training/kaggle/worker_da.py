"""lltDomainAdaptation Kaggle worker: one GPU, a list of jobs, one result file per job.
Usage: python worker_da.py <jobs.json> <out_dir>   (CUDA_VISIBLE_DEVICES picks the GPU before this starts)
Needs dalab.py and extend_tokenizer.py next to it and /kaggle/working/data/*.jsonl from the setup cell."""
import gc, json, os, sys, time, traceback, platform
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import dalab as L

jobs = json.load(open(sys.argv[1])); OUT = sys.argv[2]
DATA = "/kaggle/working/data"
rd = lambda n: [json.loads(l) for l in open(DATA + "/" + n + ".jsonl", encoding="utf-8")]
hi_test, en_edu, hi_train = rd("hi_test"), rd("en_edu"), rd("hi_train")
split = json.load(open(DATA + "/qa_split.json", encoding="utf-8"))
contexts, qa = L.load_xquad_hi()
byid = {x["id"]: x for x in qa}
distract = [t for t in (L.cut(d["text"], 1000) for d in hi_test[200:1000]) if t]
pool = contexts + distract
import hashlib
assert hashlib.sha256("|".join(pool).encode()).hexdigest()[:16] == split["poolSha"], "pool differs from the local index"
base_tok = AutoTokenizer.from_pretrained(L.MODEL)
SMOKE = any(j.get("smoke") for j in jobs)
nE = (40, 40, 60, 60, 100) if SMOKE else (200, 200, 1511, 1511, 2000)
E = {"hi": L.eval_texts(hi_test[:400], 1000, nE[0], base_tok), "en": L.eval_texts(en_edu[:500], 3000, nE[1], base_tok),
     "xsc_hi": L.load_xstorycloze("hi")[:nE[2]], "xsc_en": L.load_xstorycloze("en")[:nE[3]], "hellaswag": L.load_hellaswag(nE[4])}
Equick = {"hi": E["hi"][:64], "en": E["en"][:64]}
Q = {"test": [byid[i] for i in split["testIds"]][: (16 if SMOKE else 10 ** 6)], "train": [byid[i] for i in split["trainIds"]],
     "shots": split["shots"], "pool": pool, "contexts": contexts, "retrieved": split["retrieved"]}
ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)}
for m in ("peft", "bitsandbytes", "datasets", "tokenizers", "accelerate"):
    try:
        ENV[m] = __import__(m).__version__
    except Exception:
        pass
print("eval sizes", {k: len(v) for k, v in E.items()}, "qa test", len(Q["test"]), flush=True)


def load(path, tok_dir=None):
    tok = AutoTokenizer.from_pretrained(tok_dir or path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map="cuda:0")
    model.eval()
    return model, tok


def save(res, tag):
    json.dump(res, open(OUT + "/res_" + tag + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def job_panel(j):
    model, tok = load(j["model"])
    t = time.time()
    r = {"panel": L.panel(model, tok, E), "panelSec": round(time.time() - t, 1)}
    if j.get("qa"):
        t = time.time(); r["qa"] = L.qa_eval(model, tok, Q); r["qaSec"] = round(time.time() - t, 1)
    r["params"] = sum(p.numel() for p in model.parameters())
    return r


def job_cpt(j):
    model, tok = load(L.MODEL)
    r = {}
    if j.get("tokenizer"):
        from extend_tokenizer import extend
        t = time.time()
        tok_dir, rep = extend(L.MODEL, [d["text"] for d in hi_train[:20000]], j["new_tokens"], j["mark_regex"], OUT + "/tok_" + j["tag"])
        rep["buildSec"] = round(time.time() - t, 1); r["tokenizer"] = rep
        tok = AutoTokenizer.from_pretrained(tok_dir)
        parts = json.load(open(tok_dir + "/new_token_parts.json"))
        if j["init"] == "mean":
            L.add_tokens_mean_init(model, len(tok), parts)
        else:
            model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)    # library default: mean_resizing=True
        r["vocabRows"] = model.get_input_embeddings().weight.shape[0]
    r["panel0"] = L.panel(model, tok, E)
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # training rows: Hindi web docs with the 240 XQuAD paragraphs inserted 4 times each, plus English replay rows
    import random
    docs = [d["text"] for d in hi_train[:4000]]
    rng = random.Random(1234)
    for c in contexts * 4:
        docs.insert(rng.randrange(0, 1000), c)
    n_rows = j["steps"] * j["rows_per_step"]
    n_en = round(n_rows * j["replay"])
    hi_rows = L.pack_rows(docs, tok, 1024, n_rows - n_en)
    en_rows = L.pack_rows([d["text"] for d in en_edu[500:]], tok, 1024, max(n_en, 1))
    rows, n_en = L.mix_rows(hi_rows, en_rows, n_rows, j["replay"], seed=j.get("seed", 0))
    r["rows"] = {"total": n_rows, "english": n_en, "tokens": n_rows * 1024}
    on_eval = lambda m: {"hiBpb": round(L.bits_per_byte(m, tok, Equick["hi"])["bpb"], 5), "enBpb": round(L.bits_per_byte(m, tok, Equick["en"])["bpb"], 5)}
    cfg = dict(steps=j["steps"], rows_per_step=j["rows_per_step"], micro=j["micro"], peak_lr=j["peak_lr"],
               eval_every=j["eval_every"], warmup_frac=j.get("warmup_frac", 0.04), init_scale=j.get("init_scale", 65536.0))
    r["config"] = cfg
    log, peak, sec = L.train_cpt(model, rows, cfg, on_eval)
    r.update(metrics=log, peakMemGB=peak, trainSec=sec, tokensPerSec=round(n_rows * 1024 / sec, 1))
    r["panelFinal"] = L.panel(model, tok, E)
    if j.get("qa"):
        r["qa"] = L.qa_eval(model, tok, Q)
    ft_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if j.get("save"):
        model.half().save_pretrained(OUT + "/ckpt_" + j["tag"], safe_serialization=True)
        tok.save_pretrained(OUT + "/ckpt_" + j["tag"])
        model.float()
    r["interp"] = []
    for a in j.get("interp", []):
        L.interpolate(model, base_sd, ft_sd, a)
        r["interp"].append({"a": a, **L.panel(model, tok, E)})
        print("interp", r["interp"][-1], flush=True)
    return r


def job_extinit(j):
    # Step-0 effect of the tokenizer extension with the library's default resize (mean of all rows), no training.
    from extend_tokenizer import extend
    model, tok = load(L.MODEL)
    tok_dir, rep = extend(L.MODEL, [d["text"] for d in hi_train[:20000]], j["new_tokens"], j["mark_regex"], OUT + "/tok_" + j["tag"])
    tok = AutoTokenizer.from_pretrained(tok_dir)
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
    return {"tokenizer": rep, "vocabRows": model.get_input_embeddings().weight.shape[0], "panel0": L.panel(model, tok, E)}


def job_qlora(j):
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    src = j["model"]
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, quantization_config=q, dtype=torch.float16, device_map="cuda:0")
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    tr = [p for p in model.parameters() if p.requires_grad]
    r = {"trainable": sum(p.numel() for p in tr), "total": sum(p.numel() for p in model.parameters())}
    model.eval(); model.config.use_cache = True
    with model.disable_adapter():
        r["panelBase4bit"] = L.panel(model, tok, E)
        r["qaBase4bit"] = L.qa_eval(model, tok, Q, modes=("closed0", "rag0", "rag3"))
    ex = L.qa_train_examples(tok, Q, seed=0)
    bs = j["batch"]
    log, peak, sec = L.train_lora_qa(model, tok, ex, epochs=j["epochs"], bs=bs, micro=j.get("micro", 2), lr=j["lr"],
                                     max_steps=6 if j.get("smoke") else None)
    steps = len(log)
    r.update(metrics=log, steps=steps, trainSec=sec, peakMemGB=peak, trainExamples=len(ex), epochs=j["epochs"], batch=bs, lr=j["lr"])
    r["panel"] = L.panel(model, tok, E)
    r["qa"] = L.qa_eval(model, tok, Q, modes=("closed0", "rag0", "rag3"))
    return r


for j in jobs:
    t = time.time()
    res = {"tag": j["tag"], "job": j, "env": ENV, "date": time.strftime("%Y-%m-%d"),
           "evalSizes": {k: len(v) for k, v in E.items()}, "qaTest": len(Q["test"])}
    try:
        res.update({"panel": job_panel, "cpt": job_cpt, "qlora": job_qlora, "extinit": job_extinit}[j["kind"]](j)); res["ok"] = True
    except Exception as e:
        res["ok"] = False; res["error"] = type(e).__name__ + ": " + str(e)[:1500]; res["trace"] = traceback.format_exc()[-3000:]
        print(res["trace"], flush=True)
    res["wallSec"] = round(time.time() - t, 1)
    save(res, j["tag"])
    print("JOB DONE", j["tag"], res.get("ok"), res["wallSec"], flush=True)
    gc.collect(); torch.cuda.empty_cache()

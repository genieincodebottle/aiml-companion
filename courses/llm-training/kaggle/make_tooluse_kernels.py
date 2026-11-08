"""Build the lltToolUse Kaggle kernels from labs/lltToolUse/lltToolUse.ipynb (built from the page).

llt-tooluse-smoke     the notebook at toy size (12 rows per source, 24 BFCL items), the Qwen3 arm on, plus a tiny RUNS pass
llt-tooluse-notebook  the notebook end to end on one T4, plus a cell that records every figure to notebook_results.json
llt-tooluse-runs      the setup cells (steps 1 to 6 without the TRL check) plus RUNS: Qwen3 reference arm, seed 1, raw-text broken run
Usage: python make_tooluse_kernels.py
"""
import json, os, copy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltToolUse")
nb0 = json.load(open(os.path.join(LAB, "lltToolUse.ipynb"), encoding="utf-8"))
starter = open(os.path.join(LAB, "steps", "challenge1_starter.py"), encoding="utf-8").read().rstrip("\n")
solution = open(os.path.join(LAB, "steps", "challenge1_solution.py"), encoding="utf-8").read().rstrip("\n")


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def swap_solution(nb):
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and src(c).rstrip("\n") == starter:
            c["source"] = solution
            n += 1
    assert n == 1, "starter cell not found"


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


RECORD = r'''
# Kaggle validation only: record every figure the page quotes.
import json, platform, transformers, trl, peft, datasets, accelerate
def pack(rows_, outs, sec=None):
    return {"genSec": sec, "summary": C.summary(rows_), "outputs": outs, "correct": [bool(r["correct"]) for r in rows_],
            "parsed": [bool(r["parsed"]) for r in rows_], "officialCorrect": [bool(official(it, r["calls"])) for it, r in zip(items, rows_)]}
res = {"env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
               "datasets": datasets.__version__, "accelerate": accelerate.__version__, "gpu": torch.cuda.get_device_name(0)},
       "check": {"labelsEqual": bool(labels_equal), "own": total / count, "trl": lib, "labelledTokens": count},
       "items": [it["id"] for it in items], "categories": [it["category"] for it in items],
       "base": pack(base, base_out, base_sec), "sft": pack(sft, sft_out, sft_sec),
       "baseForced": pack(base_forced, base_forced_out, None), "sftForced": pack(sft_forced, sft_forced_out, None), "forcedSec": forced_sec,
       "pairedBootstrap": C.paired_bootstrap([r["correct"] for r in base], [r["correct"] for r in sft])}
if not USE_OUR_ADAPTER:
    res["train"] = {"sec": train_sec, "peakGB": train_peak, "steps": trainer.state.global_step,
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "loss": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]}
json.dump(res, open("/kaggle/working/notebook_results.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in res.items() if k in ("env", "check", "pairedBootstrap", "forcedSec")})
'''

RUNS = r'''
# Kaggle Run Library cell: the Qwen3 reference arm, a second seed and the raw-text broken run, with the page's own code.
import json, platform, transformers, trl, peft
from transformers import AutoModelForCausalLM
RES = {"env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "trl": trl.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)},
       "items": [it["id"] for it in items], "categories": [it["category"] for it in items], "runs": []}
official = C.official_check()
def save():
    json.dump(RES, open("/kaggle/working/tooluse_runs.json", "w"), indent=1, ensure_ascii=False)
def evaluate(model_, t, parse, constrained=False):
    t0 = time.time()
    out = C.generate(model_, t, items, constrained=constrained)
    sec = round(time.time() - t0)
    rows_ = C.score(items, out, parse)
    return {"genSec": sec, "summary": C.summary(rows_), "outputs": out, "correct": [bool(r["correct"]) for r in rows_],
            "parsed": [bool(r["parsed"]) for r in rows_], "officialCorrect": [bool(official(it, r["calls"])) for it, r in zip(items, rows_)]}
def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
def base_eval(tag, name, t, parse):
    free()
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    rec = {"tag": tag, "model": name, **evaluate(m, t, parse)}
    RES["runs"].append(rec); save()
    print(tag, rec["summary"], rec["genSec"], flush=True)
    del m
def train_eval(tag, name, t, rows_, parse, seed, alt_parse=None):
    free()
    t0 = time.time()
    tr = train_lora(name, t, rows_, seed=seed, out="o-" + tag)
    sec, peak = round(time.time() - t0), round(torch.cuda.max_memory_allocated() / 1e9, 2)
    m = tr.model.eval()
    rec = {"tag": tag, "model": name, "seed": seed, "trainSec": sec, "peakGB": peak, "steps": tr.state.global_step, "rows": len(rows_),
           "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
           "loss": [{k: h[k] for k in ("step", "loss", "grad_norm", "learning_rate") if k in h} for h in tr.state.log_history if "loss" in h],
           **evaluate(m, t, parse)}
    if alt_parse is not None:
        alt = C.score(items, rec["outputs"], alt_parse)
        rec["altParse"] = {"summary": C.summary(alt), "correct": [bool(r["correct"]) for r in alt]}
    m.save_pretrained("/kaggle/working/adapter-" + tag)
    RES["runs"].append(rec); save()
    print(tag, rec["summary"], sec, peak, flush=True)
    del tr, m

# 1. Qwen3-1.7B-Base, the previous generation, rendered in its own Hermes JSON format
REF_HEADER = ref_tok.encode("<|im_start|>assistant" + C.NL, add_special_tokens=False)
REF_END, REF_NL = ref_tok.convert_tokens_to_ids("<|im_end|>"), ref_tok.encode(C.NL, add_special_tokens=False)[0]
ref_rows = []
for c in train:
    ids = list(ref_tok.apply_chat_template(c["messages"], tools=c["tools"])["input_ids"])
    ref_rows.append({"input_ids": ids, "labels": build_labels(ids, REF_HEADER, REF_END, REF_NL)})
RES["refRowsTokens"] = sum(len(r["input_ids"]) for r in ref_rows)
base_eval("q3-1.7b-base", REF, ref_tok, C.parse_json_calls)
train_eval("q3-1.7b-lora-s0", REF, ref_tok, ref_rows, C.parse_json_calls, seed=0)
# 2. the mixed set of step 11: a third of the single-turn rows replaced by requests whose tools cannot answer
mixed_rows = []
for c in mixed:
    ids = render(c)
    mixed_rows.append({"input_ids": ids, "labels": build_labels(ids, HEADER, IM_END, NEWLINE)})
RES["mixed"] = {"negatives": len(negatives), "rows": len(mixed_rows), "example": negatives[0]["messages"][0]["content"][:300],
                "exampleTools": [t["function"]["name"] for t in negatives[0]["tools"]]}
train_eval("q35-2b-mixed-negatives", MAIN, tok, mixed_rows, parse_xml_calls, seed=0)
# 3. Qwen3.5-2B-Base, seed 1 (data order and LoRA init)
train_eval("q35-2b-lora-s1", MAIN, tok, rows, parse_xml_calls, seed=1)
# 4. broken: the Hermes conversations as raw text, their own system prompt and JSON calls, no tools argument
ROLE = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
raw_rows, skipped = [], 0
for src_rows in (single, glaive):
    kept = 0
    for row in src_rows:
        if kept == N_PER_SOURCE:
            break
        if not C.parses(row):
            continue
        chat = C.hermes_to_chat(row)
        if not chat["tools"] or len(render(chat)) > 1536:
            continue
        kept += 1
        msgs = [{"role": ROLE[t_["from"]], "content": t_["value"]} for t_ in row["conversations"]]
        ids = list(tok.apply_chat_template(msgs, chat_template=train_tpl)["input_ids"])
        if len(ids) > 2048:
            skipped += 1
            continue
        raw_rows.append({"input_ids": ids, "labels": build_labels(ids, HEADER, IM_END, NEWLINE)})
RES["rawRows"] = {"kept": len(raw_rows), "skippedOver2048": skipped, "example": tok.decode(raw_rows[0]["input_ids"])[-700:]}
train_eval("q35-2b-broken-rawtext", MAIN, tok, raw_rows, parse_xml_calls, seed=0, alt_parse=C.parse_json_calls)
print("DONE", flush=True)
'''


RUNS2 = r"""
# Kaggle Run Library cell: the Qwen3 arm again, with distinct special-token rows set before training (step 10's fix).
import json, platform, transformers, trl, peft
from transformers import AutoModelForCausalLM
from tool_common import train_lora
official = C.official_check()
RES = {"env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
               "trl": trl.__version__, "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0)},
       "items": [it["id"] for it in items], "categories": [it["category"] for it in items], "runs": []}
REF_HEADER = ref_tok.encode("<|im_start|>assistant" + C.NL, add_special_tokens=False)
REF_END, REF_NL = ref_tok.convert_tokens_to_ids("<|im_end|>"), ref_tok.encode(C.NL, add_special_tokens=False)[0]
ref_rows = []
for c in train:
    ids = list(ref_tok.apply_chat_template(c["messages"], tools=c["tools"])["input_ids"])
    ref_rows.append({"input_ids": ids, "labels": build_labels(ids, REF_HEADER, REF_END, REF_NL)})
RES["tokens"] = {s_: ref_tok.convert_tokens_to_ids(s_) for s_ in SPECIAL}
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
tr = train_lora(REF, ref_tok, ref_rows, seed=0, out="o-q3tok", special=SPECIAL)
sec, peak = round(time.time() - t0), round(torch.cuda.max_memory_allocated() / 1e9, 2)
trainable = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
loss = [{k: h[k] for k in ("step", "loss", "grad_norm", "learning_rate") if k in h} for h in tr.state.log_history if "loss" in h]
tr.model.save_pretrained("/kaggle/working/adapter-q3-1.7b-lora-rows-s0")
m = tr.model.eval()
RES["rowNormsAfterInit"] = {s_: round(m.get_input_embeddings().weight[i].float().norm().item(), 4) for s_, i in RES["tokens"].items()}
t1 = time.time()
out = C.generate(m, ref_tok, items)
gsec = round(time.time() - t1)
rows_ = C.score(items, out, C.parse_json_calls)
RES["runs"].append({"tag": "q3-1.7b-lora-rows-s0", "model": REF, "seed": 0, "trainSec": sec, "peakGB": peak, "steps": tr.state.global_step,
                    "rows": len(ref_rows), "trainable": trainable, "loss": loss, "genSec": gsec, "summary": C.summary(rows_), "outputs": out,
                    "correct": [bool(r["correct"]) for r in rows_], "parsed": [bool(r["parsed"]) for r in rows_],
                    "officialCorrect": [bool(official(it, r["calls"])) for it, r in zip(items, rows_)]})
json.dump(RES, open("/kaggle/working/tooluse_runs2.json", "w"), indent=1, ensure_ascii=False)
print(RES["runs"][0]["summary"], sec, peak, trainable, flush=True)
print("DONE", flush=True)
"""


def write_kernel(slug, title, nb):
    out = os.path.join(HERE, slug)
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, slug + ".ipynb"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    meta = {"id": "genieincodebottle/" + slug, "title": title, "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [], "keywords": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
    print("built", slug, len(nb["cells"]), "cells")


def shrink(nb, need_experiment=True):
    rep = [("N_PER_SOURCE = 600", "N_PER_SOURCE = 12"),
           ('PLAN = {"simple_python": 4, "multiple": 4, "parallel": 4, "parallel_multiple": 4, "irrelevance": 4}',
            'PLAN = {"simple_python": 50, "multiple": 50, "parallel": 50, "parallel_multiple": 50, "irrelevance": 60}'),
           ("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")]
    hits = {a: 0 for a, _ in rep}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = src(c)
        for a, b in rep:
            if a in s:
                hits[a] += 1
                s = s.replace(a, b)
        c["source"] = s
    assert all(v for k, v in hits.items() if need_experiment or k != "RUN_EXPERIMENT = False"), hits


SKIP_IN_RUNS = ("labels_equal", "base_out = C.generate", "USE_OUR_ADAPTER", "sft_forced_out", "RUN_EXPERIMENT", "raise NotImplementedError",
                "All checks passed")

if __name__ == "__main__":
    nb = copy.deepcopy(nb0)
    swap_solution(nb)
    nb["cells"].append(code_cell(RECORD))
    write_kernel("llt-tooluse-notebook", "LLT ToolUse Notebook", nb)

    sm = copy.deepcopy(nb0)
    swap_solution(sm)
    shrink(sm)
    idx = [i for i, c in enumerate(sm["cells"]) if c["cell_type"] == "code" and "RUN_EXPERIMENT = True" in src(c)][0]
    sm["cells"].insert(idx, code_cell(RECORD))
    sm["cells"].append(code_cell("from tool_common import train_lora"))
    sm["cells"].append(code_cell(RUNS))
    write_kernel("llt-tooluse-smoke", "LLT ToolUse Smoke", sm)

    rn = copy.deepcopy(nb0)
    keep = [rn["cells"][0]]
    for c in rn["cells"][1:]:
        s = src(c)
        if c["cell_type"] == "code" and "def make_negatives" in s:
            keep.append(c)                        # RUN_EXPERIMENT stays False there; RUNS trains the mixed set itself
        elif c["cell_type"] == "code" and not any(k in s for k in SKIP_IN_RUNS):
            keep.append(c)
    keep.append(code_cell("from tool_common import train_lora"))
    keep.append(code_cell(RUNS))
    rn["cells"] = keep
    if "--runs-smoke" in sys.argv:
        shrink(rn, need_experiment=False)
    write_kernel("llt-tooluse-runs", "LLT ToolUse Runs", rn)

    r2 = copy.deepcopy(nb0)
    keep = [r2["cells"][0]]
    for c in r2["cells"][1:]:
        s_ = src(c)
        if c["cell_type"] == "code" and "SPECIAL = [" in s_:
            keep.append(c)                        # step 10 with RUN_EXPERIMENT off: the embedding-row check runs
        elif c["cell_type"] == "code" and (s_.startswith("%%writefile") or (not any(k in s_ for k in SKIP_IN_RUNS) and "def make_negatives" not in s_)):
            keep.append(c)
    keep.append(code_cell(RUNS2))
    r2["cells"] = keep
    write_kernel("llt-tooluse-q3tok", "LLT ToolUse Q3tok", r2)

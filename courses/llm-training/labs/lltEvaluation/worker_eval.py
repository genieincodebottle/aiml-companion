"""Experiment queue for kernel llt-eval-main (runs on GPU 1 beside the notebook, or after it on a one-GPU session).

It reuses the page cells' functions: the imports, function definitions and a few named constants of each cell are
executed (ast filter); the cells' own runs are not. Jobs:
  extra    lm-eval panel + five ARC formats for SmolLM2-360M, SmolLM2-360M-Instruct, Qwen3-0.6B-Base, Qwen3.5-2B-Base, Qwen3.5-2B
  seed1    benign LoRA on Qwen3-1.7B with seed 1, then the behaviour panel, IFEval and ARC formats
  safety   the same LoRA with 10% refusal examples mixed in (refusals written by Qwen3-1.7B itself to hh-rlhf red-team prompts)
  gsm8k    why lm-eval's gsm8k stalls on Qwen3 base models, with a 5-minute cap, then the fixed run
Usage: python worker_eval.py <cells dir> <out dir> <job,job,...>
"""
import ast, gc, json, os, subprocess, sys, time, traceback
from types import SimpleNamespace

CELLS, OUT, JOBS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("LLT_SMOKE") == "1"
ns = {"__name__": "worker", "SMOKE": SMOKE}

def load_defs(fname, keep=()):
    tree = ast.parse(open(os.path.join(CELLS, fname), encoding="utf-8").read())
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in keep for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), fname, "exec"), ns)

load_defs("c01_harness.py", ("TM",))
import torch, numpy as np
ns.update(device="cuda", NL=chr(10), DTYPE="float16", LIMIT=16 if SMOKE else 1000, TASKS=["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"],
          MMLU_PER_SUBJECT=2 if SMOKE else 20, np=np, torch=torch)
for f, keep in [("c02_scorer.py", ()), ("c03_formats.py", ("LETTERS", "arc")), ("c04_ci.py", ()), ("c05_paired.py", ()),
                ("c06_generative.py", ("ifeval", "N_IF", "N_GEN")), ("c07_contamination.py", ("dolly",)),
                ("c08_judge.py", ("XS", "xs_prompts", "labelled", "human", "JUDGE_ASK")), ("c09_behaviour.py", ("N_SYC", "PUSHBACK", "xs")),
                ("c10_sft.py", ("SFT_STEPS",))]:
    load_defs(f, keep)
if SMOKE:
    ns["labelled"] = ns["labelled"].sample(64, random_state=0).reset_index(drop=True)
    ns["human"] = ns["labelled"].final_label.str.startswith(("2_", "3_")).to_numpy()
    ns["xs"] = ns["xs_prompts"].groupby("label").head(12)
TUNED = "Qwen/Qwen3-1.7B"
ENV = {"python": sys.version.split()[0], "torch": torch.__version__}
for pkg in ["transformers", "peft", "lm_eval", "datasets", "accelerate", "numpy"]:
    from importlib.metadata import version
    ENV[pkg] = version(pkg.replace("_", "-") if pkg == "lm_eval" else pkg)

def save(name, obj):
    obj = dict(obj, env=ENV, gpu=torch.cuda.get_device_name(0), date=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

def free():
    gc.collect()
    torch.cuda.empty_cache()

def load_tuned():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TUNED)
    model = AutoModelForCausalLM.from_pretrained(TUNED, dtype=torch.float16).to("cuda").eval()
    return model, tok

def calibrate_judge(model, tok):
    # Same procedure as page cell c08: threshold picked on even rows of the XSTest human labels.
    replies64 = [ns["first_tokens"](tok, r) for r in ns["labelled"].completion]
    J = ns["judge_scores"](model, tok, ns["labelled"].prompt.tolist(), replies64)
    fit = (np.arange(len(J)) % 2) == 0
    grid = np.linspace(-6, 6, 121)
    ns["THRESH"] = float(grid[np.argmax([np.mean((J[fit] > g) == ns["human"][fit]) for g in grid])])
    sm = np.array([ns["strmatch"](r) for r in replies64])
    agree_j = np.mean((J[~fit] > ns["THRESH"]) == ns["human"][~fit])
    ns["CLASSIFIER"] = "judge" if agree_j > np.mean(sm[~fit] == ns["human"][~fit]) else "string match"
    return ns["THRESH"]

def after_panels(pm, tok, log, extra):
    t = time.time()
    beh = ns["behaviour_panel"](pm, tok, judge_off=pm.disable_adapter)
    ife = ns["ifeval_scores"](pm, tok)
    fm = ns["score_formats"](SimpleNamespace(model=pm, tokenizer=tok))
    return dict(extra, behaviour=beh, ifeval=ife, formats={f: v.tolist() for f, v in fm.items()}, sft=log, panelSeconds=round(time.time() - t, 1))

def job_extra():
    models = ["HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B"]
    for name in models[:2] if SMOKE else models:
        t = time.time()
        try:
            torch.cuda.reset_peak_memory_stats()
            lm, out = ns["run_harness"](name)
            ft = time.time()
            out["formats"] = {f: v.tolist() for f, v in ns["score_formats"](lm).items()}
            out["formatsSeconds"] = round(time.time() - ft, 1)
            out["peakGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            out.pop("arcRequests", None)
            del lm
        except Exception as e:
            out = {"model": name, "error": type(e).__name__ + ": " + str(e)[:800], "trace": traceback.format_exc()[-2000:]}
        out["wallSec"] = round(time.time() - t, 1)
        save("extra_" + name.split("/")[-1], out)
        print("extra", name, out.get("seconds"), out.get("error", ""), flush=True)
        free()

def job_seed1():
    model, tok = load_tuned()
    thr = calibrate_judge(model, tok)
    torch.cuda.reset_peak_memory_stats()
    pm, log = ns["train_lora"](model, tok, ns["dolly_rows"](1), 1)
    save("arm_seed1", after_panels(pm, tok, log, {"arm": "benign LoRA, seed 1", "threshold": thr}))
    del pm, model
    free()

def job_safety():
    import gzip
    from huggingface_hub import hf_hub_download
    model, tok = load_tuned()
    thr = calibrate_judge(model, tok)
    path = hf_hub_download("Anthropic/hh-rlhf", "harmless-base/train.jsonl.gz", repo_type="dataset", revision="09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa")
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    NL = chr(10)
    prompts = []
    for r in rows[:64 if SMOKE else 800]:
        first = r["chosen"].split(NL + NL + "Human: ")[1].split(NL + NL + "Assistant:")[0].strip()
        if 10 <= len(first) <= 400:
            prompts.append(first)
    t = time.time()
    outs = ns["generate"](model, tok, [ns["chat"](tok, [{"role": "user", "content": p}]) for p in prompts], max_new=160, batch_size=64)
    j = ns["judge_scores"](model, tok, prompts, [ns["first_tokens"](tok, o) for o, _ in outs])
    keep = [(p, o.strip()) for p, (o, ended), s in zip(prompts, outs, j) if s > thr and ended]
    n_mix = 8 if SMOKE else 200
    refusals = [{"prompt": p, "response": o} for p, o in keep[:n_mix]]       # texts stay in memory only
    gen_sec = round(time.time() - t, 1)
    print("safety data | prompts", len(prompts), "| refused and ended", len(keep), "| mixed in", len(refusals), "|", gen_sec, "s", flush=True)
    data = ns["dolly_rows"](0)[:2000 - len(refusals)] + refusals
    torch.cuda.reset_peak_memory_stats()
    pm, log = ns["train_lora"](model, tok, data, 0)
    save("arm_safety_mix", after_panels(pm, tok, log, {"arm": "benign LoRA + 10% refusals, seed 0", "threshold": thr,
                                                     "safetyData": {"source": "Anthropic/hh-rlhf harmless-base train, first human turns",
                                                                    "prompts": len(prompts), "refusedAndEnded": len(keep), "mixed": len(refusals),
                                                                    "generateSeconds": gen_sec}}))
    del pm, model
    free()

def job_gsm8k():
    res = {}
    code = (
        "import json, sys, time, lm_eval" + chr(10) +
        "from lm_eval.models.huggingface import HFLM" + chr(10) +
        "lm = HFLM(pretrained=sys.argv[1], dtype='float16', batch_size=16, device='cuda')" + chr(10) +
        "print('generation_config.max_new_tokens =', lm.model.generation_config.max_new_tokens, flush=True)" + chr(10) +
        "if sys.argv[2] == 'fix': lm.model.generation_config.max_new_tokens = None" + chr(10) +
        "t = time.time()" + chr(10) +
        "out = lm_eval.simple_evaluate(model=lm, tasks=['gsm8k'], num_fewshot=8, limit=int(sys.argv[3]), log_samples=True)" + chr(10) +
        "s = [x for x in out['samples']['gsm8k'] if x.get('filter', 'strict-match') == 'strict-match']" + chr(10) +
        "toks = [len(lm.tokenizer(x['resps'][0][0])['input_ids']) for x in s]" + chr(10) +
        "r = out['results']['gsm8k']" + chr(10) +
        "json.dump({'seconds': round(time.time() - t, 1), 'n': len(s), 'strict': r['exact_match,strict-match'], 'flexible': r['exact_match,flexible-extract'],"
        " 'strictPerItem': [x['exact_match'] if 'exact_match' in x else None for x in s], 'meanReplyTokens': sum(toks) / len(toks), 'maxReplyTokens': max(toks)}, open(sys.argv[4], 'w'))" + chr(10))
    open(os.path.join(OUT, "gsm8k_run.py"), "w").write(code)
    for tag, model, mode, n, cap in [("asis-0.6b", "Qwen/Qwen3-0.6B-Base", "asis", 16, 90 if SMOKE else 300), ("fix-0.6b", "Qwen/Qwen3-0.6B-Base", "fix", 16, 300),
                                     ("fix-1.7b-250", "Qwen/Qwen3-1.7B-Base", "fix", 8 if SMOKE else 250, 900)]:
        rf = os.path.join(OUT, "gsm8k_" + tag + ".json")
        t = time.time()
        try:
            p = subprocess.run([sys.executable, os.path.join(OUT, "gsm8k_run.py"), model, mode, str(n), rf], capture_output=True, text=True, timeout=cap)
            tail = (p.stdout or "")[-1500:] + (p.stderr or "")[-2500:]
        except subprocess.TimeoutExpired as e:
            tail = "TIMEOUT after " + str(cap) + " s. " + ((e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else str(e.stdout or ""))[-800:]
        r = json.load(open(rf)) if os.path.exists(rf) else {}
        r.update(model=model, mode=mode, limit=n, capSec=cap, wallSec=round(time.time() - t, 1), tail=tail)
        res[tag] = r
        print("gsm8k", tag, r.get("seconds"), r.get("strict"), r.get("meanReplyTokens"), "TIMEOUT" in tail, flush=True)
        save("gsm8k_diag", {"runs": res})

for j in JOBS:
    t = time.time()
    try:
        {"extra": job_extra, "seed1": job_seed1, "safety": job_safety, "gsm8k": job_gsm8k}[j]()
    except Exception:
        traceback.print_exc()
        save("error_" + j, {"trace": traceback.format_exc()[-4000:]})
    print("JOB", j, "done in", round(time.time() - t), "s", flush=True)
    free()
print("WORKER DONE", flush=True)

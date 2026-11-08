CELLS_SRC = r'''
import json, os, subprocess, sys, time, platform, re, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def sh(cmd, timeout=900, cwd=None):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return dict(cmd=cmd[:200], rc=p.returncode, sec=round(time.time() - t, 1), tail=(p.stdout or "")[-1800:] + (p.stderr or "")[-1800:])
    except subprocess.TimeoutExpired:
        return dict(cmd=cmd[:200], rc=-9, sec=round(time.time() - t, 1), tail="TIMEOUT")
OUT = {"date": time.strftime("%Y-%m-%d"), "steps": []}
def save():
    json.dump(OUT, open("/kaggle/working/llt_c_probe_f.json", "w"), indent=1)
def step(name, cmd, timeout=900, cwd=None):
    r = sh(cmd, timeout, cwd); r["name"] = name; OUT["steps"].append(r); save(); print(name, r["rc"], r["sec"], flush=True); print(r["tail"][-600:], flush=True); return r
def ver():
    r = sh(sys.executable + " -c \"import torch,transformers;print(torch.__version__,transformers.__version__)\"")
    return r["tail"].strip().splitlines()[-1] if r["tail"].strip() else "?"
step("uninstall torchao", sys.executable + " -m pip uninstall -y -q torchao")
step("pin hf stack", sys.executable + " -m pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2 datasets==5.1.0")
OUT["baseVersions"] = ver(); save()
# lm-eval extras for ifeval
step("install lm-eval + ifeval extras", sys.executable + " -m pip install -q lm-eval==0.4.13 langdetect immutabledict nltk", 600)
step("nltk data", sys.executable + " -c \"import nltk; print(nltk.download('punkt'), nltk.download('punkt_tab'))\"", 300)
OUT["afterLmEval"] = ver(); save()
LM = "python -m lm_eval --model hf --model_args pretrained=%s,dtype=float16 --tasks %s --batch_size %s --limit %s --output_path /kaggle/working/lm_%s --seed 0 --log_samples"
step("lm-eval ifeval Qwen3-0.6B limit100 maxgen256", LM % ("Qwen/Qwen3-0.6B", "ifeval", "16", "100", "q3ifeval") + " --apply_chat_template --gen_kwargs max_gen_toks=256", 420)
step("lm-eval gsm8k Qwen3-0.6B-Base limit32 8shot maxgen256", LM % ("Qwen/Qwen3-0.6B-Base", "gsm8k", "8", "32", "q3gsm") + " --num_fewshot 8 --gen_kwargs max_gen_toks=256", 420)
step("lm-eval arc_easy+hellaswag Qwen3-0.6B-Base fp16 limit200 (transformers 5.22.1)", LM % ("Qwen/Qwen3-0.6B-Base", "arc_easy,hellaswag", "16", "200", "q3base_a"), 300)
step("install llmcompressor", sys.executable + " -m pip install -q llmcompressor==0.15.0", 900)
OUT["afterLlmCompressor"] = ver(); save()
GPTQ = """
import time, torch, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
name = "Qwen/Qwen3-0.6B-Base"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda:0")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2000]")
ds = ds.filter(lambda e: len(e["text"]) > 200).select(range(128))
def prep(e): return tok(e["text"], max_length=512, truncation=True, add_special_tokens=False)
ds = ds.map(prep, remove_columns=ds.column_names)
t = time.time(); torch.cuda.reset_peak_memory_stats()
oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]), max_seq_length=512, num_calibration_samples=128)
sec = time.time() - t
model.save_pretrained("/kaggle/working/q3-0.6b-gptq", save_compressed=True); tok.save_pretrained("/kaggle/working/q3-0.6b-gptq")
sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk("/kaggle/working/q3-0.6b-gptq") for f in fs)
print("RESULT " + json.dumps(dict(gptqSec=round(sec, 1), peakGB=round(torch.cuda.max_memory_allocated() / 1e9, 2), outBytes=sz)))
"""
open("/kaggle/working/gptq.py", "w").write(GPTQ)
step("llm-compressor GPTQ W4A16", sys.executable + " /kaggle/working/gptq.py 2>&1 | grep -E 'RESULT|Error|error' | tail -5", 600)
step("lm-eval arc_easy+hellaswag Qwen3-0.6B-Base fp16 limit200 (after llmcompressor)", LM % ("Qwen/Qwen3-0.6B-Base", "arc_easy,hellaswag", "16", "200", "q3base_b"), 300)
step("lm-eval arc_easy+hellaswag GPTQ W4A16 limit200", LM % ("/kaggle/working/q3-0.6b-gptq", "arc_easy,hellaswag", "16", "200", "q3gptq"), 400)
OUT["totalMinutes"] = round((time.time() - T0) / 60, 1)
save(); print("DONE", OUT["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe F: lm-eval ifeval and gsm8k, GPTQ quality"), ("code", "import time\nT0 = time.time()\n" + CELLS_SRC)]

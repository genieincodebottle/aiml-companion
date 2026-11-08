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
    json.dump(OUT, open("/kaggle/working/llt_c_probe_c.json", "w"), indent=1)
def step(name, cmd, timeout=900, cwd=None):
    r = sh(cmd, timeout, cwd); r["name"] = name; OUT["steps"].append(r); save(); print(name, r["rc"], r["sec"], flush=True); print(r["tail"][-600:], flush=True); return r
def ver():
    r = sh(sys.executable + " -c \"import torch,transformers;print(torch.__version__,transformers.__version__)\"")
    return r["tail"].strip().splitlines()[-1] if r["tail"].strip() else "?"
step("uninstall torchao", sys.executable + " -m pip uninstall -y -q torchao")
step("pin hf stack", sys.executable + " -m pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 bitsandbytes==0.50.2 datasets==5.1.0")
OUT["baseVersions"] = ver(); save()
# 1. resolver dry runs: what would each library pull?
for lib in ["lm-eval==0.4.13", "lighteval==0.13.0", "mergekit==0.1.5", "llmcompressor==0.15.0"]:
    step("dry-run " + lib, sys.executable + " -m pip install --dry-run -q " + lib + " 2>&1 | grep -i -E 'would install|error|conflict' | head -5", 600)
# 2. lm-eval
step("install lm-eval", sys.executable + " -m pip install -q lm-eval==0.4.13", 600)
OUT["afterLmEval"] = ver(); save()
LM = "python -m lm_eval --model hf --model_args pretrained=%s,dtype=float16 --tasks %s --batch_size %s --limit %s --output_path /kaggle/working/lm_%s --seed 0"
step("lm-eval arc_easy+hellaswag Qwen3-0.6B-Base limit200", LM % ("Qwen/Qwen3-0.6B-Base", "arc_easy,hellaswag", "16", "200", "q3base"), 900)
step("lm-eval arc_easy+hellaswag SmolLM2-360M limit200", LM % ("HuggingFaceTB/SmolLM2-360M", "arc_easy,hellaswag", "16", "200", "smol360"), 900)
step("lm-eval ifeval Qwen3-0.6B limit100", LM % ("Qwen/Qwen3-0.6B", "ifeval", "16", "100", "q3ifeval") + " --apply_chat_template", 900)
step("lm-eval gsm8k Qwen3-0.6B-Base limit100 8shot", LM % ("Qwen/Qwen3-0.6B-Base", "gsm8k", "16", "100", "q3gsm") + " --num_fewshot 8", 1200)
# 3. llm-compressor GPTQ W4A16
r = step("install llmcompressor", sys.executable + " -m pip install -q llmcompressor==0.15.0", 900)
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
print(json.dumps(dict(gptqSec=round(sec, 1), peakGB=round(torch.cuda.max_memory_allocated() / 1e9, 2), outBytes=sz)))
"""
open("/kaggle/working/gptq.py", "w").write(GPTQ)
step("llm-compressor GPTQ W4A16 Qwen3-0.6B-Base", sys.executable + " /kaggle/working/gptq.py", 1200)
# 4. llama.cpp GGUF
import urllib.request
rel = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1"))[0]
tag = rel["tag_name"]; OUT["llamaCppTag"] = tag
asset = [a for a in rel["assets"] if re.match(r"^llama-b\d+-bin-ubuntu-x64\.tar\.gz$", a["name"])]
OUT["llamaAsset"] = asset[0]["name"] if asset else None; OUT["llamaAssetNames"] = [a["name"] for a in rel["assets"] if "ubuntu" in a["name"]][:20]; save()
if asset:
    step("llama.cpp binaries", "mkdir -p /kaggle/working/lcpp && cd /kaggle/working/lcpp && curl -sL -o b.tgz %s && tar xzf b.tgz && ls" % asset[0]["browser_download_url"], 600)
step("llama.cpp source", "cd /kaggle/working && curl -sL -o src.tgz https://github.com/ggml-org/llama.cpp/archive/refs/tags/%s.tar.gz && tar xzf src.tgz && ls" % tag, 600)
src = glob.glob("/kaggle/working/llama.cpp-*")
if src:
    S = src[0]
    step("gguf deps", sys.executable + " -m pip install -q sentencepiece protobuf; ls %s/requirements | head" % S, 600)
    step("save HF model", sys.executable + " -c \"from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B-Base', local_dir='/kaggle/working/q3base'))\"", 600)
    step("convert f16", "PYTHONPATH=%s/gguf-py %s %s/convert_hf_to_gguf.py /kaggle/working/q3base --outtype f16 --outfile /kaggle/working/q3.f16.gguf" % (S, sys.executable, S), 900)
    bins = glob.glob("/kaggle/working/lcpp/**/llama-quantize", recursive=True); pp = glob.glob("/kaggle/working/lcpp/**/llama-perplexity", recursive=True)
    OUT["bins"] = dict(quantize=bins[:1], perplexity=pp[:1]); save()
    if bins:
        bd = os.path.dirname(bins[0])
        for q in ["Q8_0", "Q4_K_M", "Q2_K"]:
            step("quantize " + q, "LD_LIBRARY_PATH=%s %s /kaggle/working/q3.f16.gguf /kaggle/working/q3.%s.gguf %s && ls -l /kaggle/working/q3.%s.gguf" % (bd, bins[0], q, q, q), 600)
        step("wikitext raw test", "cd /kaggle/working && python -c \"from datasets import load_dataset; d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test'); open('wt2.test.raw','w').write(''.join(d['text']))\"", 300)
        if pp:
            for q in ["f16", "Q8_0", "Q4_K_M", "Q2_K"]:
                step("perplexity " + q, "LD_LIBRARY_PATH=%s %s -m /kaggle/working/q3.%s.gguf -f /kaggle/working/wt2.test.raw --chunks 20 2>&1 | tail -4" % (bd, pp[0], q), 900)
OUT["totalMinutes"] = round((time.time() - T0) / 60, 1)
save(); print("DONE", OUT["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe C: lm-eval, llm-compressor, llama.cpp on a T4\n\nWrites `llt_c_probe_c.json`."), ("code", "import time\nT0 = time.time()\n" + CELLS_SRC)]

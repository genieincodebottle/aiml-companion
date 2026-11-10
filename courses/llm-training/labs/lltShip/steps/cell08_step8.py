import json, site, venv
from safetensors.torch import load_file

def side_env(path, packages, share=True):
    # A virtual environment for a tool that pins other versions. With share=True it also sees this runtime's
    # packages, torch included, but its own installs come first on its path.
    if not os.path.isdir(path):
        venv.create(path, with_pip=True)
        if share:
            purelib = sh([path + "/bin/python", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])[1].strip()
            open(purelib + "/runtime.pth", "w").write(NL.join(site.getsitepackages()))
        sh([path + "/bin/python", "-m", "pip", "install", "-q", *packages])

side_env("lc-env", ["llmcompressor==0.15.0"])          # llm-compressor 0.15.0 pins transformers 5.21.0

GPTQ = '''import json, sys, time, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from safetensors.torch import save_file
calib, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained("merged")
model = AutoModelForCausalLM.from_pretrained("merged", dtype=torch.float16, device_map="cuda:0")
rows = json.load(open(calib))
ds = Dataset.from_dict(dict(tok(rows, max_length=512, truncation=True)))
t = time.time()
oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        max_seq_length=512, num_calibration_samples=len(rows))
sec = time.time() - t
model.save_pretrained(out, save_compressed=True)
tok.save_pretrained(out)
# Read the saved 4-bit checkpoint back, unpacked to fp16 weights, so the main notebook can score exactly what ships
from transformers import CompressedTensorsConfig
back = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float16, quantization_config=CompressedTensorsConfig(run_compressed=False))
keep = {k: v.contiguous() for k, v in back.state_dict().items()
        if k != "lm_head.weight" and not k.endswith(("_scale", "_shape", "_zero_point", "_g_idx"))}
save_file(keep, out + ".fp16.safetensors")       # weights on the 4-bit grid, stored as fp16; scales dropped
print("RESULT", json.dumps({"oneshotSec": round(sec, 1)}))
'''
open("gptq.py", "w").write(GPTQ)

# Two calibration sets of 128 rows: English Wikipedia text, or text like what this model must keep doing
wiki = [r["text"] for r in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:3000]") if len(r["text"]) > 200][:128]
hi_train = [r["text"][:2000] for r in load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", streaming=True).take(64)]
uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=1).select(range(64))
own = hi_train + [TEMPLATE.format(ex["prompt"]) + ex["chosen"][-1]["content"].strip() + "<|endoftext|>" for ex in uf]
for name, rows in [("wikitext", wiki), ("own", own)]:
    json.dump(rows, open("calib-" + name + ".json", "w"))
    sec, out = sh(["lc-env/bin/python", "gptq.py", "calib-" + name + ".json", "gptq-" + name])
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("gptq-" + name) for f in fs if f.endswith(".safetensors"))
    load_into(load_file("gptq-" + name + ".fp16.safetensors"))
    key = "gptq " + name
    runs[key] = panel(model)
    runs[key]["MiB"], runs[key]["sec"] = round(size / 2 ** 20, 1), json.loads(out.split("RESULT ")[1].splitlines()[0])["oneshotSec"]
    show("GPTQ W4A16, " + name, runs[key])
    print("   ", runs[key]["MiB"], "MiB of weights | GPTQ took", runs[key]["sec"], "s | Hindi bpb vs merged",
          ci(runs[best], runs[key], "hi"), "| reply NLL vs merged", ci(runs[best], runs[key], "reply"))
print("own minus wikitext calibration, Hindi bpb", ci(runs["gptq wikitext"], runs["gptq own"], "hi"),
      "| reply NLL", ci(runs["gptq wikitext"], runs["gptq own"], "reply"))
load_into(merged_sd)

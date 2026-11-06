"""Cells of the learner notebook lltQLoRA.ipynb. build_notebook.py builds the notebook from this list,
and the Kaggle measurement kernel (kaggle/cells_qlora_lab.py) runs the same cells with recorder cells added.

Kinds: "md" (markdown), "code" (a lab-page code block, shipped), "rec" (Kaggle kernel only: records results).
Code cells hold no backslashes, backticks or dollar-brace, because the lab page quotes them in JS template literals.
"""

CELLS = [
("md", """# QLoRA: fine-tune a 9B model on a free GPU

LLM Training Lab, module 13. Regime PEFT: the 4-bit base stays frozen, only LoRA adapters train, objective masked NLL on the answer tokens.

Run the cells in order on one T4 (Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2).
The first run downloads about 19 GB for Qwen3.5-9B-Base and 1.6 GB for Qwen3.5-0.8B-Base."""),

("code", '''import os, sys, platform, shutil, subprocess
from importlib.metadata import version, PackageNotFoundError
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # one GPU, set before torch loads; Trainer refuses a model split over Kaggle's two T4s
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
problems = []
if sys.version_info < (3, 11):
    problems.append("Python 3.11 or newer is needed (3.12 recommended). Start a current Colab or Kaggle session.")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:                                   # a CPU runtime has no nvidia-smi
    smi = []
gpu = smi[0] if smi else ""
if not gpu:
    problems.append("No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    problems.append("A P100 has no kernels in current PyTorch. Pick a T4.")
elif int(gpu.split(",")[1].split()[0]) < 14000:
    problems.append("This GPU has under 14 GB of memory; the 9B needs a 15-16 GB card such as a T4.")
cache = os.path.expanduser("~/.cache")                    # where Hugging Face stores the downloads
os.makedirs(cache, exist_ok=True)
free_disk = shutil.disk_usage(cache).free / 1e9
if free_disk < 25:
    problems.append("Free at least 25 GB of disk for the two downloads (" + str(round(free_disk)) + " GB free).")
try:
    torch_version = version("torch")
except PackageNotFoundError:
    torch_version = "not installed"
print("Python", platform.python_version(), "| GPU", gpu or "none", "| torch", torch_version, "| disk free", round(free_disk), "GB")
for p in problems:
    print("PROBLEM", p)
print("Runtime OK" if not problems else "Fix the PROBLEM lines, then run this cell again.")'''),

("code", '''# Pinned Hugging Face stack of our recorded run. torch and numpy stay as the runtime ships them.
# Kaggle ships torchao 0.10, which PEFT 0.21 refuses next to LoRA layers, so remove it.
%pip uninstall -y -q torchao
%pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0 optuna==5.0.0'''),

("code", '''import torch
CAP = torch.cuda.get_device_capability(0)
free_gb = torch.cuda.mem_get_info()[0] / 1e9
assert free_gb > 14, "The 9B needs a whole 15-16 GB card. Restart the runtime to free the GPU."
# torch.cuda.is_bf16_supported() is True on a T4, which has no bf16 tensor cores, so test the capability instead.
DTYPE = torch.bfloat16 if CAP >= (8, 0) else torch.float16
print(torch.__version__, torch.cuda.get_device_name(0), CAP, round(free_gb, 1), "GB free |", DTYPE)'''),

("rec", '''import json, time
import transformers, trl, peft, bitsandbytes, accelerate, datasets, optuna
R = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count(),
     "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
             "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__,
             "datasets": datasets.__version__, "optuna": optuna.__version__}, "cellSec": {}}
T_START = time.time()
def save():
    R["elapsedSec"] = round(time.time() - T_START, 1)
    json.dump(R, open("/kaggle/working/llt_qlora_lab.json", "w"), indent=1, default=str)
save(); print(R)'''),

# Step 2. NF4 by hand on real weights
("code", '''import json, struct, time
import numpy as np, requests
from scipy.stats import norm

URL = "https://huggingface.co/Qwen/Qwen3.5-9B-Base/resolve/main/"

def read_tensor(name):
    # A safetensors file starts with an 8-byte header length and a JSON header of byte ranges.
    # Three HTTP range requests fetch one tensor without the 19 GB download.
    shard = URL + requests.get(URL + "model.safetensors.index.json").json()["weight_map"][name]
    n = struct.unpack("<Q", requests.get(shard, headers={"Range": "bytes=0-7"}).content)[0]
    info = json.loads(requests.get(shard, headers={"Range": "bytes=8-" + str(7 + n)}).content)[name]
    a, b = info["data_offsets"]
    raw = requests.get(shard, headers={"Range": "bytes=" + str(8 + n + a) + "-" + str(7 + n + b)}).content
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16    # bf16 is the top half of a float32
    return bits.view(np.float32).reshape(info["shape"])

W = read_tensor("model.language_model.layers.14.mlp.down_proj.weight")
print(W.shape, "std", round(float(W.std()), 5), "absmax", round(float(np.abs(W).max()), 4))'''),

("code", '''def nf4_levels(offset=0.9677083):
    # 8 positive and 7 negative quantiles of N(0, 1), plus an exact zero, scaled into [-1, 1].
    pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
    neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    v = np.sort(np.concatenate([neg, [0.0], pos]))
    return v / v.max()

def quantize(w, levels, block=64):
    x = w.reshape(-1, block)
    absmax = np.abs(x).max(axis=1, keepdims=True)          # one scale per block of 64 weights
    mid = (levels[1:] + levels[:-1]) / 2
    codes = np.searchsorted(mid, x / absmax)                # index of the nearest level, 0 to 15
    return codes, absmax, (levels[codes] * absmax).reshape(w.shape)

def rel_err(w, wq):
    return float(np.linalg.norm(w.astype(np.float64) - wq) / np.linalg.norm(w.astype(np.float64)))

NF4 = nf4_levels()
INT4 = np.arange(-7, 8) / 7                                 # uniform symmetric int4, 15 levels
print("NF4 levels", np.round(NF4, 4))
for name, lv in [("NF4", NF4), ("int4", INT4)]:
    errs = [round(100 * rel_err(W, quantize(W, lv, b)[2]), 2) for b in (64, 4096, W.shape[1], W.size)]
    print(name, "relative error % at block 64, 4096, one row, whole tensor:", errs)'''),

("code", '''import bitsandbytes.functional as F

t = torch.from_numpy(W).cuda()
q, state = F.quantize_4bit(t, blocksize=64, quant_type="nf4")
packed = q.cpu().numpy().ravel()
bnb_codes = np.stack([packed >> 4, packed & 15], axis=1).ravel()      # two 4-bit codes per byte
ours = quantize(W, NF4)[0].ravel()
print("codes equal to bitsandbytes:", float((bnb_codes == ours).mean()), "of", ours.size)
q2, state2 = F.quantize_4bit(t, blocksize=64, quant_type="nf4", compress_statistics=True)
nbytes = lambda x: x.numel() * x.element_size()
single = (nbytes(q) + nbytes(state.absmax)) * 8 / W.size
double = (nbytes(q2) + nbytes(state2.absmax) + nbytes(state2.state2.absmax)) * 8 / W.size
print("bits per weight: single quant", round(single, 3), "| double quant", round(double, 3), "| absmax stored as", state2.absmax.dtype)
del t, q, q2'''),

("rec", '''R["nf4"] = {"shape": list(W.shape), "codesEqual": float((bnb_codes == ours).mean()), "bitsSingle": single, "bitsDouble": double,
            "relErrNF4": rel_err(W, quantize(W, NF4)[2]), "relErrInt4": rel_err(W, quantize(W, INT4)[2])}
save()'''),

# Step 3. Load the 9B in 4-bit
("code", '''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-9B-Base"
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",      # the default is fp4
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, dtype=DTYPE, device_map={"": 0})
tok = AutoTokenizer.from_pretrained(MODEL)
print(type(model).__name__, "loaded in", round(time.time() - t0), "s, download included")
print("GPU memory after load", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
n4 = sum(p.numel() for p in model.parameters() if p.dtype == torch.uint8)
n16 = sum(p.numel() for p in model.parameters() if p.dtype != torch.uint8)
print("4-bit weights", 2 * n4, "| 16-bit weights", n16, "| numel() reports", n4 + n16)'''),

("rec", '''R["load"] = {"sec": round(time.time() - t0, 1), "class": type(model).__name__, "memGB": torch.cuda.memory_allocated() / 1e9,
             "footprintGB": model.get_memory_footprint() / 1e9, "packedBytes": n4, "params16": n16,
             "eos": tok.eos_token, "pad": tok.pad_token, "padSide": tok.padding_side}
save()'''),

# Step 4. Attach LoRA the T4-safe way
("code", '''from peft import LoraConfig, get_peft_model

# prepare_model_for_kbit_training would cast the two 248,320 x 4,096 embedding matrices to fp32 and run out of memory.
for p in model.parameters():
    p.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False

def add_lora(m, r=16, dora=False):
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear",
                     use_dora=dora, task_type="CAUSAL_LM")
    m = get_peft_model(m, cfg)
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # fp32 adapters, so the fp16 GradScaler can unscale their gradients
    return m

model = add_lora(model)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
names = sorted({n.split(".lora_A")[0].split(".")[-1] for n, p in model.named_parameters() if "lora_A" in n})
print(trainable, "trainable parameters in", len(names), "kinds of layer:", names)'''),

("rec", '''R["lora"] = {"trainable": trainable, "targets": names, "memGB": torch.cuda.memory_allocated() / 1e9}
save()'''),

# Step 5. Memory and speed by sequence length
("code", '''import statistics
from datasets import load_dataset

wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tok(" ".join(t for t in wiki["text"][:20000] if t.strip())[:900000], return_tensors="pt").input_ids[0]
trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable_params, lr=0.0)          # lr 0: measure only, the adapters stay at their start values
scaler = torch.amp.GradScaler("cuda", init_scale=2048)

def measure(seq, batch, steps=4):
    model.train()                                          # gradient checkpointing only runs in training mode
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        x = ids[s * batch * seq:(s + 1) * batch * seq].view(batch, seq).cuda()
        t = time.time()
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(input_ids=x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.time() - t)
    return statistics.median(times[1:]), torch.cuda.max_memory_allocated() / 1e9

sweep = {}
for seq, batch in [(512, 1), (512, 4), (1024, 1), (1024, 2), (2048, 1), (2048, 2)]:
    try:
        sec, gb = measure(seq, batch)
        sweep[(seq, batch)] = (sec, gb)
        print(seq, "x", batch, "|", round(sec, 2), "s/step |", round(seq * batch / sec), "tokens/s | peak", round(gb, 2), "GB")
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        print(seq, "x", batch, "| out of memory")
del opt, scaler'''),

("rec", '''R["sweep"] = {f"{k[0]}x{k[1]}": {"sPerStep": v[0], "tokPerSec": k[0] * k[1] / v[0], "peakGB": v[1]} for k, v in sweep.items()}
save()'''),

# Step 6. Data and the base model's score
("code", '''from datasets import load_dataset

bank = load_dataset("PolyAI/banking77")
NAMES = bank["train"].features["label"].names
PROMPT = """Classify the intent of this banking customer message.
Message: {text}
Intent:"""

def to_row(ex):
    return {"prompt": PROMPT.format(text=ex["text"]), "completion": " " + NAMES[ex["label"]]}

pool = bank["train"].shuffle(seed=0)
dev = pool.select(range(500)).map(to_row, remove_columns=["text", "label"])
train = pool.select(range(500, 3500)).map(to_row, remove_columns=["text", "label"])
test = bank["test"].map(to_row, remove_columns=["text", "label"])
print(len(NAMES), "intents |", len(train), "train,", len(dev), "dev,", len(test), "test")
print(train[0]["prompt"] + train[0]["completion"])'''),

("code", '''@torch.no_grad()
def accuracy(m, rows, tk, batch=64, prefix=""):
    tk.padding_side = "left"                              # generate after the prompt, not after padding
    m.eval(); hits = valid = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tk([prefix + p for p in chunk["prompt"]], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=DTYPE):
            out = m.generate(**enc, max_new_tokens=16, do_sample=False, use_cache=True, pad_token_id=tk.pad_token_id)
        for gold, o in zip(chunk["completion"], out[:, enc["input_ids"].shape[1]:]):
            words = tk.decode(o, skip_special_tokens=True).split()
            pred = words[0] if words else ""
            hits += pred == gold.strip(); valid += pred in NAMES
    tk.padding_side = "right"
    return hits / len(rows), valid / len(rows)'''),

("code", '''LISTED = """Possible intents: {names}.

""".format(names=", ".join(NAMES))
t0 = time.time()
with model.disable_adapter():
    base_acc, base_valid = accuracy(model, test.select(range(500)), tok, batch=32, prefix=LISTED)
print("base 9B with the intent list in the prompt: accuracy", round(base_acc, 3), "| valid intent names", round(base_valid, 3),
      "|", round(time.time() - t0), "s")'''),

("rec", '''R["base"] = {"acc": base_acc, "valid": base_valid, "n": 500, "sec": round(time.time() - t0, 1),
             "promptTokens": len(tok(LISTED + test[0]["prompt"]).input_ids)}
save()'''),

# Step 7. Train with SFTTrainer
("code", '''from trl import SFTConfig, SFTTrainer

def sft_args(out, lr=2e-4, epochs=1, warmup=0.03, seed=0, save_steps=50):
    return SFTConfig(output_dir=out, per_device_train_batch_size=16, learning_rate=lr, num_train_epochs=epochs,
                     warmup_steps=warmup, lr_scheduler_type="linear", max_length=256, completion_only_loss=True,
                     fp16=DTYPE == torch.float16, bf16=DTYPE == torch.bfloat16, gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10,
                     save_steps=save_steps, save_only_model=False, report_to="none", seed=seed, data_seed=seed)

def make_trainer(m, args, data, tk, callbacks=None):
    tr = SFTTrainer(model=m, args=args, train_dataset=data, processing_class=tk, callbacks=callbacks)
    # SFTTrainer casts LoRA weights of a 4-bit model to bf16. The fp16 GradScaler cannot unscale bf16 gradients.
    recast = [n for n, p in tr.model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    for p in tr.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return tr, recast

trainer, recast = make_trainer(model, sft_args("qlora-9b"), train, tok)
print(len(recast), "trainable tensors were re-cast by SFTTrainer, e.g.", recast[:1])'''),

("code", '''import torch.nn.functional as Fn

batch = {k: v.cuda() for k, v in next(iter(trainer.get_train_dataloader())).items() if torch.is_tensor(v)}
with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
    library = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss.float()
labels = batch["labels"][:, 1:]
mask = labels != -100                                       # prompt and padding tokens carry no loss
own = Fn.cross_entropy(logits[:, :-1][mask], labels[mask])
print("own masked NLL", round(own.item(), 5), "| library", round(library.item(), 5),
      "| tokens with loss", int(mask.sum()), "of", labels.numel(), "| batch", list(batch["input_ids"].shape))
del logits'''),

("rec", '''R["lossCheck"] = {"own": own.item(), "library": library.item(), "absDiff": abs(own.item() - library.item()),
                  "lossTokens": int(mask.sum()), "tokens": labels.numel(), "batchShape": list(batch["input_ids"].shape),
                  "recastCount": len(recast), "recastExample": recast[:3],
                  "lastLabelOfExample": int(batch["labels"][0][int(batch["attention_mask"][0].sum()) - 1]),
                  "eosId": tok.eos_token_id,
                  "lastTokensOfExample": tok.convert_ids_to_tokens(batch["input_ids"][0][batch["attention_mask"][0].bool()][-4:].tolist())}
save()'''),

("code", '''torch.cuda.reset_peak_memory_stats()
t0 = time.time()
result = trainer.train()
train_sec = time.time() - t0
log = [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]
print(trainer.state.global_step, "steps in", round(train_sec), "s |", round(train_sec / trainer.state.global_step, 2), "s/step | peak",
      round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
print("loss every 10 steps", log)
trainer.save_model("qlora-9b/final")                       # the adapter only, not the 9B
print(sorted(os.listdir("qlora-9b")), round(os.path.getsize("qlora-9b/final/adapter_model.safetensors") / 1e6), "MB adapter")'''),

("rec", '''R["train"] = {"steps": trainer.state.global_step, "sec": train_sec, "sPerStep": train_sec / trainer.state.global_step,
              "peakGB": torch.cuda.max_memory_allocated() / 1e9, "log": trainer.state.log_history,
              "adapterMB": os.path.getsize("qlora-9b/final/adapter_model.safetensors") / 1e6,
              "ckptFiles": sorted(os.listdir("qlora-9b/checkpoint-50"))}
save()'''),

("code", '''REPO, SUB = "aimlcompanion/llm-training-lab", "lltQLoRA/qlora-9b-final"   # our step 8 adapter, 173 MB
USE_OUR_CHECKPOINT = False      # True: skip code cell 15 and score our trained adapter in code cell 17
if USE_OUR_CHECKPOINT:
    try:
        model.load_adapter(REPO, adapter_name="ours", subfolder=SUB)
        model.set_adapter("ours")
        print("Loaded our adapter from", REPO + "/" + SUB)
    except Exception as e:
        print("Adapter not reachable:", type(e).__name__, "- keep your own adapter from code cell 15.")
else:
    print("Skipped. Set USE_OUR_CHECKPOINT = True to load our trained adapter instead of training.")'''),

("code", '''t0 = time.time()
ft_acc, ft_valid = accuracy(model, test, tok)
print("fine-tuned 9B: accuracy", round(ft_acc, 4), "| valid intent names", round(ft_valid, 4), "| on", len(test),
      "test messages in", round(time.time() - t0), "s")'''),

("rec", '''R["ft"] = {"acc": ft_acc, "valid": ft_valid, "n": len(test), "evalSec": round(time.time() - t0, 1)}
save()'''),

# Step 8. Resume after a session expiry
("code", '''import gc
from safetensors.torch import load_file

assert os.path.exists("qlora-9b/checkpoint-150"), "This step resumes from the checkpoints of code cell 15. You skipped cell 15, so skip step 9."
for name in ("trainer", "model", "batch", "trainable_params"):
    globals().pop(name, None)                               # free what earlier cells made
gc.collect(); torch.cuda.empty_cache()
print("GPU memory after freeing", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")

# A new session: reload the 4-bit base, add fresh adapters, and resume from the last checkpoint at step 150.
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, dtype=DTYPE, device_map={"": 0})
for p in model.parameters():
    p.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads(); model.config.use_cache = False
model = add_lora(model)
trainer, _ = make_trainer(model, sft_args("qlora-9b"), train, tok)
trainer.train(resume_from_checkpoint="qlora-9b/checkpoint-150")
resumed = [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]
print("resumed run, loss every 10 steps", resumed)
a = load_file("qlora-9b/final/adapter_model.safetensors")
trainer.save_model("qlora-9b/resumed")
b = load_file("qlora-9b/resumed/adapter_model.safetensors")
print("largest adapter difference, uninterrupted vs resumed:", max((a[k].float() - b[k].float()).abs().max().item() for k in a))'''),

("rec", '''R["resume"] = {"log": trainer.state.log_history,
               "maxAbsDiff": max((a[k].float() - b[k].float()).abs().max().item() for k in a),
               "maxAbs": max(a[k].float().abs().max().item() for k in a), "tensors": len(a)}
save()'''),

# Step 9. Tune on a budget with a proxy
("code", '''import gc, optuna
import torch.nn.functional as Fn
from transformers import TrainerCallback

for name in ("trainer", "model"):
    globals().pop(name, None)
gc.collect(); torch.cuda.empty_cache()
print("GPU memory after freeing", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
PROXY = "Qwen/Qwen3.5-0.8B-Base"
ptok = AutoTokenizer.from_pretrained(PROXY)

def load_proxy(quant=True, r=16, dora=False, seed=0):
    torch.manual_seed(seed)
    kw = {"quantization_config": bnb_cfg} if quant else {}
    m = AutoModelForCausalLM.from_pretrained(PROXY, dtype=DTYPE, device_map={"": 0}, **kw)
    for p in m.parameters():
        p.requires_grad = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads(); m.config.use_cache = False
    return add_lora(m, r=r, dora=dora)

@torch.no_grad()
def dev_loss(m, rows, tk, batch=8):
    # Mean NLL of the answer tokens, the number a validation-loss curve shows.
    m.eval(); tot = n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        p_ids = [tk(p).input_ids for p in chunk["prompt"]]
        full = [tk(p + c).input_ids + [tk.eos_token_id] for p, c in zip(chunk["prompt"], chunk["completion"])]
        L = max(len(f) for f in full)
        x = torch.full((len(full), L), tk.pad_token_id); y = torch.full((len(full), L), -100); am = torch.zeros(len(full), L, dtype=torch.long)
        for j, (p, f) in enumerate(zip(p_ids, full)):
            x[j, :len(f)] = torch.tensor(f); am[j, :len(f)] = 1; y[j, len(p):len(f)] = torch.tensor(f[len(p):])
        with torch.autocast("cuda", dtype=DTYPE):
            logits = m(input_ids=x.cuda(), attention_mask=am.cuda()).logits.float()
        lbl = y[:, 1:].cuda(); keep = lbl != -100
        tot += Fn.cross_entropy(logits[:, :-1][keep], lbl[keep], reduction="sum").item(); n += int(keep.sum())
    return tot / n

class EpochReport(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
    def on_epoch_end(self, args, state, control, model=None, **kw):
        acc = accuracy(model, dev, ptok)[0]
        self.trial.set_user_attr("dev_loss_" + str(round(state.epoch)), dev_loss(model, dev, ptok))
        model.train()
        self.trial.report(acc, round(state.epoch))
        if self.trial.should_prune():
            raise optuna.TrialPruned()'''),

("rec", '''big = []
for o in gc.get_objects():
    try:
        if torch.is_tensor(o) and o.is_cuda and o.numel() * o.element_size() > 1e8:
            big.append((round(o.numel() * o.element_size() / 1e9, 3), str(o.dtype), list(o.shape), [type(r).__name__ for r in gc.get_referrers(o)][:4]))
    except Exception:
        pass
R["cudaLeftAfterFree"] = {"allocatedGB": torch.cuda.memory_allocated() / 1e9, "bigTensors": sorted(big, reverse=True)[:12]}
save(); print(R["cudaLeftAfterFree"])'''),

("code", '''def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 1, 3)
    rank = trial.suggest_categorical("rank", [4, 8, 16, 32, 64])
    warmup = trial.suggest_float("warmup", 0.0, 0.2)
    m = load_proxy(r=rank)
    tr = None
    try:
        tr, _ = make_trainer(m, sft_args("tune/" + str(trial.number), lr, epochs, warmup, save_steps=10000), train, ptok,
                             callbacks=[EpochReport(trial)])
        t = time.time(); tr.train(); trial.set_user_attr("train_sec", time.time() - t)
        acc = accuracy(m, dev, ptok)[0]
        trial.set_user_attr("dev_loss", dev_loss(m, dev, ptok))
        trial.set_user_attr("test_acc", accuracy(m, test, ptok)[0])
        return acc
    finally:
        del tr, m                                             # free the GPU before the next trial, pruned or not
        gc.collect(); torch.cuda.empty_cache()

USE_OUR_STUDY = False      # True: download our 8-trial journal and run no new trials
STUDY_URL = "https://raw.githubusercontent.com/genieincodebottle/aiml-companion/main/courses/llm-training/runs/lltQLoRA/tune.log"
if USE_OUR_STUDY and not os.path.exists("tune.log"):
    r = requests.get(STUDY_URL, timeout=60)
    try:
        r.raise_for_status()                                # a 404 stops here, before tune.log is written
    except requests.HTTPError as err:
        raise RuntimeError("Our journal is not reachable (" + str(err) + "). Set USE_OUR_STUDY = False to run the search, "
                           "or read our trial table on the lab page.") from err
    open("tune.log", "wb").write(r.content)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    study_name="qlora-proxy", direction="maximize",
    sampler=optuna.samplers.RandomSampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0),
    storage=optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend("tune.log")),
    load_if_exists=True)                                    # a new session picks up where the log stops
n_done = len(study.trials)
if n_done == 0:
    study.enqueue_trial({"lr": 2e-4, "epochs": 1, "rank": 16, "warmup": 0.03})   # the common QLoRA default, as trial 0
t0 = time.time()
study.optimize(objective, n_trials=0 if USE_OUR_STUDY else 16 - n_done, timeout=3600,           # 16 trials or one hour, whichever comes first
               catch=(torch.OutOfMemoryError,))                       # a trial that runs out of memory is marked FAIL
print("tuning took", round(time.time() - t0), "s")
for t in study.trials:
    print(t.number, t.state.name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.params.items()},
          "dev acc", t.value and round(t.value, 3), "dev loss", round(t.user_attrs.get("dev_loss", float("nan")), 3))'''),

("rec", '''R["tune"] = {"sec": time.time() - t0, "trials": [{"number": t.number, "state": t.state.name, "params": t.params, "value": t.value,
              "intermediate": t.intermediate_values, "attrs": t.user_attrs,
              "durationSec": t.duration.total_seconds() if t.duration else None} for t in study.trials]}
save()'''),

("code", '''from scipy.stats import spearmanr

done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
best = study.best_trial
print("best trial", best.number, best.params, "dev acc", round(best.value, 3), "test acc", round(best.user_attrs["test_acc"], 4))
print("trial 0 (default)", study.trials[0].params, "dev acc", round(study.trials[0].value, 3),
      "test acc", round(study.trials[0].user_attrs["test_acc"], 4))
print("importance, PED-ANOVA (the Optuna 5 default):", optuna.importance.get_param_importances(study))
try:
    print("importance, fANOVA:", optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0)))
except RuntimeError as err:                                 # every complete trial scored the same
    print("fANOVA skipped:", err)
rho = spearmanr([t.user_attrs["dev_loss"] for t in done], [t.value for t in done]).statistic
by_loss = min(done, key=lambda t: t.user_attrs["dev_loss"])
print(len(done), "complete trials | Spearman(dev loss, dev accuracy)", round(rho, 3),
      "| lowest dev loss is trial", by_loss.number, "with dev acc", round(by_loss.value, 3))'''),

("rec", '''R["tuneSummary"] = {"best": best.number, "bestParams": best.params, "bestDev": best.value, "bestTest": best.user_attrs["test_acc"],
                     "pedAnova": optuna.importance.get_param_importances(study),
                     "spearmanLossAcc": rho, "lowestLossTrial": by_loss.number, "complete": len(done)}
save()'''),

# Step 10. NF4 vs 16-bit LoRA vs DoRA on the proxy
("code", '''arms = {"QLoRA (NF4 base)": dict(quant=True), "LoRA (16-bit base)": dict(quant=False), "QLoRA + DoRA": dict(quant=True, dora=True)}
bake = []
for arm, kw in arms.items():
    for seed in (1, 2):
        m = load_proxy(r=best.params["rank"], seed=seed, **kw)
        tr, _ = make_trainer(m, sft_args("arm", best.params["lr"], best.params["epochs"], best.params["warmup"], seed=seed,
                                         save_steps=10000), train, ptok)
        torch.cuda.reset_peak_memory_stats(); t = time.time(); tr.train(); sec = time.time() - t
        row = dict(arm=arm, seed=seed, test_acc=accuracy(m, test, ptok)[0], s_per_step=sec / tr.state.global_step,
                   peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        bake.append(row); print(row)
        del tr, m; gc.collect(); torch.cuda.empty_cache()'''),

("rec", '''R["arms"] = bake
save()'''),

# Step 11. Apply the tuned settings to the 9B
("code", '''model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, dtype=DTYPE, device_map={"": 0})
for p in model.parameters():
    p.requires_grad = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads(); model.config.use_cache = False
model = add_lora(model, r=best.params["rank"])
trainer, _ = make_trainer(model, sft_args("qlora-9b-tuned", best.params["lr"], best.params["epochs"], best.params["warmup"],
                                          save_steps=10000), train, tok)
t0 = time.time(); trainer.train(); tuned_sec = time.time() - t0
tuned_acc = accuracy(model, test, tok)[0]
print("9B with the proxy's best settings:", best.params, "| test accuracy", round(tuned_acc, 4),
      "| default settings", round(ft_acc, 4), "|", trainer.state.global_step, "steps in", round(tuned_sec), "s")'''),

("rec", '''R["tuned9b"] = {"params": best.params, "acc": tuned_acc, "sec": tuned_sec, "steps": trainer.state.global_step,
                "log": trainer.state.log_history, "peakGB": torch.cuda.max_memory_allocated() / 1e9}
save()'''),
("rec", '''import shutil, glob
for d in glob.glob("qlora-9b/checkpoint-*") + ["tune", "arm", "qlora-9b-tuned", "qlora-9b/resumed"]:
    shutil.rmtree(d, ignore_errors=True)
R["done"] = True; save(); print("kept", os.listdir("."), os.listdir("qlora-9b"))'''),
]

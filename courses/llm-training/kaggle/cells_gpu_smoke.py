CELLS = [
("md", "# LLM Training Lab: GPU smoke test\n\nChecks the free Kaggle GPU the training labs run on: device, versions, bf16, fp16 throughput, a short fp16 training run and a 4-bit model load. Writes `llt_gpu_smoke.json`."),
("code", r'''
import json, os, platform, subprocess, sys, time
R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
R["python"] = platform.python_version()
print("python", R["python"])
'''),
("code", r'''
# Pinned stack from the plan (versions checked 2026-11-04).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.22.1", "trl==1.17.1", "peft==0.21.2", "accelerate", "bitsandbytes", "datasets"], check=False)
'''),
("code", r'''
import torch, importlib
R["torch"] = torch.__version__
R["cuda_runtime"] = torch.version.cuda
R["gpu_count"] = torch.cuda.device_count()
R["gpus"] = [torch.cuda.get_device_name(i) for i in range(R["gpu_count"])]
R["capability"] = list(torch.cuda.get_device_capability(0))
R["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
R["bf16_supported"] = torch.cuda.is_bf16_supported()
for m in ["transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets"]:
    try:
        R[m] = importlib.import_module(m).__version__
    except Exception as e:
        R[m] = "IMPORT FAILED: " + repr(e)[:200]
print(json.dumps(R, indent=1))
'''),
("code", r'''
# Achieved matmul throughput (fp16 and fp32), the ceiling every T4 lab works under.
def tflops(dtype, n=8192, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): a @ b
    torch.cuda.synchronize()
    return round(2 * n**3 * iters / (time.time() - t) / 1e12, 1)
R["matmul_tflops_fp16"] = tflops(torch.float16)
R["matmul_tflops_fp32"] = tflops(torch.float32, n=4096)
print(R["matmul_tflops_fp16"], R["matmul_tflops_fp32"])
'''),
("code", r'''
# A short fp16 + GradScaler training run on a GPT-2-shaped model from random init.
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)
cfg = GPT2Config(n_layer=6, n_head=6, n_embd=384, n_positions=512, vocab_size=8192)
model = GPT2LMHeadModel(cfg).cuda()
R["smoke_model_params"] = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler("cuda")
B, T = 16, 512
x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
torch.cuda.reset_peak_memory_stats()
losses = []
for step in range(30):
    if step == 5: torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=x, labels=x).loss
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    losses.append(round(loss.item(), 3))
torch.cuda.synchronize()
R["smoke_train_tokens_per_sec"] = int(B * T * 25 / (time.time() - t0))
R["smoke_train_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
R["smoke_train_loss_first_last"] = [losses[0], losses[-1]]
print(R["smoke_model_params"], R["smoke_train_tokens_per_sec"], R["smoke_train_peak_mem_gb"], losses[0], losses[-1])
del model, opt; torch.cuda.empty_cache()
'''),
("code", r'''
# 4-bit NF4 load of a small open model (the QLoRA path), one forward pass.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    name = "Qwen/Qwen3-0.6B"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="cuda:0")
    R["nf4_load_sec"] = round(time.time() - t, 1)
    with torch.no_grad():
        out = m(**tok("The capital of France is", return_tensors="pt").to("cuda"))
    R["nf4_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    R["nf4_ok"] = True
except Exception as e:
    R["nf4_ok"] = False; R["nf4_error"] = repr(e)[:400]
print({k: R[k] for k in R if k.startswith("nf4")})
'''),
("code", r'''
R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open("/kaggle/working/llt_gpu_smoke.json", "w") as f: json.dump(R, f, indent=1)
print(json.dumps(R, indent=1))
'''),
]

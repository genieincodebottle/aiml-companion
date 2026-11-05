CELLS = [
("md", "# LLM Training Lab: lltAttention on a T4\n\nNaive causal attention vs PyTorch SDPA (each backend) for T = 128 to 4,096 on one Tesla T4: peak memory and time, forward and forward + backward, fp16. Also the fp16 mask-value checks and an fp16 leakage test. Writes `llt_attention_t4.json`."),
("code", r'''
import contextlib, json, math, platform, statistics, subprocess, time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
R = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
R["python"] = platform.python_version()
R["torch"] = torch.__version__
R["cuda_runtime"] = torch.version.cuda
R["gpu_count"] = torch.cuda.device_count()
R["gpu"] = torch.cuda.get_device_name(0)
R["capability"] = list(torch.cuda.get_device_capability(0))
R["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
R["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
R["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
R["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
print(json.dumps(R, indent=1))
'''),
("code", r'''
DEV, DTYPE = "cuda", torch.float16
B, H, D = 4, 12, 64            # batch 4, GPT-2 small's 12 heads of width 64
TS = [128, 256, 512, 1024, 2048, 4096]

def naive(q, k, v):
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

BACKENDS = {"sdpa_default": None, "sdpa_math": SDPBackend.MATH,
            "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION, "sdpa_flash": SDPBackend.FLASH_ATTENTION,
            "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION}

def run_once(fn, q, k, v, g, backward):
    y = fn(q, k, v)
    if backward:
        y.backward(g)
        q.grad = k.grad = v.grad = None
    return y

def measure(name, T, backward, iters=20):
    fn = naive if name == "naive" else sdpa
    backend = None if name == "naive" else BACKENDS[name]
    ctx = sdpa_kernel([backend]) if backend is not None else contextlib.nullcontext()
    torch.cuda.empty_cache()
    q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=DTYPE, requires_grad=backward) for _ in range(3))
    g = torch.randn(B, H, T, D, device=DEV, dtype=DTYPE)
    try:
        with ctx:
            for _ in range(3):
                run_once(fn, q, k, v, g, backward)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            y = run_once(fn, q, k, v, g, backward)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - base
            del y
            times = []
            for _ in range(iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                run_once(fn, q, k, v, g, backward)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        return {"peakMiB": round(peak / 2**20, 2), "ms": round(statistics.median(times), 3),
                "msMin": round(min(times), 3), "msMax": round(max(times), 3)}
    except torch.OutOfMemoryError as ex:
        return {"error": "OOM", "detail": str(ex)[:300]}
    except RuntimeError as ex:
        return {"error": "RuntimeError", "detail": str(ex)[:300]}
    finally:
        del q, k, v, g
        torch.cuda.empty_cache()

sweep = []
for backward in (False, True):
    for T in TS:
        for name in ["naive"] + list(BACKENDS):
            r = measure(name, T, backward)
            r.update({"impl": name, "T": T, "backward": backward,
                      "scoresMiB": round(B * H * T * T * 2 / 2**20, 2)})
            sweep.append(r)
            print(r)
R["config"] = {"B": B, "H": H, "D": D, "dtype": "float16", "iters": 20, "warmup": 3,
               "memory": "torch.cuda.max_memory_allocated() minus memory_allocated() before the timed call",
               "time": "median of 20 CUDA-event timings after 3 warm-up calls"}
R["sweep"] = sweep
'''),
("code", r'''
# fp16 accuracy of the naive path against SDPA (default backend), forward only
acc = []
torch.manual_seed(0)
for T in TS:
    q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=DTYPE) for _ in range(3))
    with torch.no_grad():
        a, b = naive(q, k, v), sdpa(q, k, v)
        a32 = naive(q.float(), k.float(), v.float())
    acc.append({"T": T, "maxAbsNaiveVsSdpa": (a.float() - b.float()).abs().max().item(),
                "maxAbsSdpaVsFp32": (b.float() - a32).abs().max().item(),
                "maxAbsNaiveVsFp32": (a.float() - a32).abs().max().item()})
    print(acc[-1])
R["fp16_accuracy"] = acc

# fp16 leakage test with SDPA: change token t+1, outputs at positions <= t must not move
T = 64
q, k, v = (torch.randn(1, H, T, D, device=DEV, dtype=DTYPE) for _ in range(3))
worst = 0.0
with torch.no_grad():
    y = sdpa(q, k, v)
    for t in range(T - 1):
        k2, v2 = k.clone(), v.clone()
        k2[:, :, t + 1] += torch.randn(H, D, device=DEV, dtype=DTYPE)
        v2[:, :, t + 1] += torch.randn(H, D, device=DEV, dtype=DTYPE)
        worst = max(worst, (y[:, :, : t + 1] - sdpa(q, k2, v2)[:, :, : t + 1]).abs().max().item())
R["fp16_leakage_sdpa_worst"] = worst
print("leakage worst", worst)
'''),
("code", r'''
# The -1e9 mask value in fp16
mv = {}
T = 8
q, k, v = (torch.randn(1, 1, T, D, device=DEV, dtype=DTYPE) for _ in range(3))
scores = q @ k.transpose(-2, -1) / math.sqrt(D)
future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEV), 1)
try:
    scores.masked_fill(future, -1e9)
    mv["masked_fill_minus_1e9"] = "ok"
except RuntimeError as ex:
    mv["masked_fill_minus_1e9"] = "RuntimeError: " + str(ex)
with torch.autocast("cuda", dtype=torch.float16):
    s_auto = q.float() @ k.float().transpose(-2, -1)
    mv["autocast_scores_dtype"] = str(s_auto.dtype)
    try:
        s_auto.masked_fill(future, -1e9)
        mv["autocast_masked_fill_minus_1e9"] = "ok"
    except RuntimeError as ex:
        mv["autocast_masked_fill_minus_1e9"] = "RuntimeError: " + str(ex)
add = torch.full((T, T), -1e9, device=DEV).triu(1).to(DTYPE)
mv["additive_mask_fp16_value"] = str(add[0, 1].item())
# a left-padded sequence: position 0 is padding, every key there is masked for every query,
# and query 0 sees only padding, so its whole row is masked
pad = torch.zeros(T, T, dtype=torch.bool, device=DEV); pad[:, 0] = True
row_masked = future | pad
for name, fill in [("minus_1e9_cast_to_fp16", torch.tensor(-1e9).to(DTYPE).item()),
                   ("finfo_min_fp16", torch.finfo(DTYPE).min)]:
    w = torch.softmax(scores.masked_fill(row_masked, fill), dim=-1)
    y1 = w @ v
    # second layer: attend over y1 as keys and values with a correct causal mask
    s2 = (y1 @ y1.transpose(-2, -1) / math.sqrt(D)).masked_fill(future, fill)
    y2 = torch.softmax(s2, dim=-1) @ y1
    mv[name] = {"fill": float(fill), "row0_nan": bool(torch.isnan(w[0, 0, 0]).any()),
                "layer1_nan_positions": int(torch.isnan(y1).any(-1).sum()),
                "layer2_nan_positions": int(torch.isnan(y2).any(-1).sum())}
R["mask_value_fp16"] = mv
print(json.dumps(mv, indent=1))
'''),
("code", r'''
R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open("/kaggle/working/llt_attention_t4.json", "w") as f:
    json.dump(R, f, indent=1)
print("saved")
'''),
]

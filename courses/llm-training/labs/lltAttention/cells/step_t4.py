# Needs a CUDA GPU (Colab or Kaggle T4). Naive causal attention vs SDPA, float16, batch 4, 12 heads of width 64.
import math, statistics
import torch
import torch.nn.functional as F

def naive(q, k, v):
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def measure(fn, T, backward, B=4, h=12, d=64, iters=20):
    q, k, v = (torch.randn(B, h, T, d, device="cuda", dtype=torch.float16, requires_grad=backward) for _ in range(3))
    g = torch.randn_like(q)                     # the gradient arriving from the layer above
    def call():
        y = fn(q, k, v)
        if backward:
            y.backward(g)
            q.grad = k.grad = v.grad = None     # count the gradients inside every call
    for _ in range(3):
        call()                                  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    call()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return peak, statistics.median(times)

if not torch.cuda.is_available():
    print("No CUDA GPU on this runtime, so this cell stops here. Read the table on the lab page,",
          "or switch to a T4 GPU (Colab: Runtime > Change runtime type) and run it again.")
else:
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__)
    for backward in (False, True):
        print("forward + backward" if backward else "forward only")
        print("     T   naive MiB   SDPA MiB   naive ms   SDPA ms")
        for T in [128, 256, 512, 1024, 2048, 4096]:
            (nm, nt), (sm, st) = measure(naive, T, backward), measure(sdpa, T, backward)
            print(f"{T:6d} {nm:11.1f} {sm:10.1f} {nt:10.2f} {st:9.2f}")

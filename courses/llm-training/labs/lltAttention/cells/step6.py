# Peak memory of one causal attention call on the CPU, naive vs PyTorch SDPA.
from torch.profiler import profile, ProfilerActivity

def naive(q, k, v):
    T = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # (B, h, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def peak_mib(fn, *args):
    # Replay every allocation and free the profiler recorded, in time order, and keep the peak.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn(*args)
    events = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events()
                    if e.name() == "[memory]")
    live = peak = 0
    for _, nbytes in events:
        live += nbytes
        peak = max(peak, live)
    return peak / 2**20

B, h, d = 1, 12, 64                      # one sequence, GPT-2 small's 12 heads of width 64
memory = {}
print("     T   T x T scores   naive peak   SDPA peak   naive / SDPA   (MiB)")
for T in [128, 256, 512, 1024, 2048, 4096]:
    q, k, v = (torch.randn(B, h, T, d) for _ in range(3))
    scores = B * h * T * T * 4 / 2**20     # float32 is 4 bytes
    n, s = peak_mib(naive, q, k, v), peak_mib(sdpa, q, k, v)
    memory[T] = (scores, n, s)
    print(f"{T:6d} {scores:13.1f} {n:12.1f} {s:11.1f} {n / s:13.1f}")

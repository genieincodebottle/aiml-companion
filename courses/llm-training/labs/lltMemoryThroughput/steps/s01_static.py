import torch
from gpt import GPT, GPTConfig

cap = torch.cuda.get_device_capability()
dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16   # pick by capability, not by is_bf16_supported()
print(torch.cuda.get_device_name(), cap, "| is_bf16_supported():", torch.cuda.is_bf16_supported(), "| using", dtype)

torch.manual_seed(0)
model = GPT(GPTConfig()).cuda()
N = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
x = torch.randint(0, 50257, (1, 64), device="cuda")
for _ in range(2):
    with torch.autocast("cuda", dtype=dtype):
        loss = model(x, x)
    scaler.scale(loss).backward()      # no zero_grad: params, grads and Adam state all stay alive
    scaler.step(opt)
    scaler.update()

GB = 1e9
alloc = torch.cuda.memory_allocated()
print(f"parameters {N:,}")
print(f"allocated after a step {alloc / GB:.3f} GB = {alloc / N:.2f} bytes per parameter")
print(f"16 bytes per parameter {16 * N / GB:.3f} GB")
print(f"reserved by the caching allocator {torch.cuda.memory_reserved() / GB:.3f} GB")
del model, opt, scaler, loss, x   # loss.grad_fn still points at every parameter and its grad
torch.cuda.empty_cache()

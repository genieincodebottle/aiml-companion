# fsdp.py - launch with: torchrun --nproc_per_node=2 fsdp.py  (or 1, to see one T4 run out of memory)
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from gpt import GPT, GPTConfig

CFG = GPTConfig(layers=24, heads=16, d=2048)          # about 1.3B parameters
GB = 1e9

def build():
    torch.manual_seed(0)
    model = GPT(CFG, ckpt=True, chunks=8)              # built full size on the CPU
    mp = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mp_policy=mp)               # each block is gathered, used and freed as one unit
    fully_shard(model, mp_policy=mp)                   # embeddings, final norm and the tied head
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, opt

def train(model, opt, scaler, steps, gen, B=2, S=1024):
    for _ in range(steps):
        x = torch.randint(0, CFG.vocab, (B, S), device="cuda", generator=gen)
        loss = model(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    return loss.item()

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    model, opt = build()
    N = sum(p.numel() for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator(device="cuda").manual_seed(100 + rank)
    try:
        train(model, opt, scaler, 2, gen)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        loss = train(model, opt, scaler, 5, gen)
        torch.cuda.synchronize()
        tps = 5 * 2 * 1024 * world / (time.perf_counter() - t)
        print(f"rank {rank} of {world}  N {N:,}  peak {torch.cuda.max_memory_allocated() / GB:.2f} GB  "
              f"16N/{world} = {16 * N / world / GB:.2f} GB  {tps:,.0f} tokens/s  loss {loss:.3f}")
    except torch.cuda.OutOfMemoryError:
        print(f"rank {rank} of {world}  N {N:,}  OUT OF MEMORY at {torch.cuda.memory_allocated() / GB:.2f} GB allocated, "
              f"16N = {16 * N / GB:.2f} GB")
    dist.destroy_process_group()

# allreduce.py - launch with: torchrun --nproc_per_node=2 allreduce.py
import time
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, n = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
for mib in (1, 16, 64, 256, 512):
    x = torch.ones(mib * 2**20 // 4, device="cuda")      # mib MiB of fp32
    for _ in range(3):
        dist.all_reduce(x)                                # warm-up, sets up the NCCL channels
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 10
    algbw = x.numel() * 4 / dt / 1e9                      # message bytes per second
    busbw = algbw * 2 * (n - 1) / n                       # bytes each GPU actually sent per second
    if rank == 0:
        print(f"{mib:>4} MiB  {1e3 * dt:7.2f} ms  algbw {algbw:5.2f} GB/s  busbw {busbw:5.2f} GB/s")
dist.destroy_process_group()

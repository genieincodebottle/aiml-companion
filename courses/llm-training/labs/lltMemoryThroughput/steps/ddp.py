# ddp.py - launch with: torchrun --nproc_per_node=2 ddp.py [--sync-every-micro] [--same-seed]
import contextlib, sys, time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, GPTConfig

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
torch.manual_seed(0)                                   # same init everywhere; DDP also broadcasts rank 0's weights
model = DDP(GPT(GPTConfig()).cuda(), device_ids=[rank])
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
scaler = torch.amp.GradScaler("cuda")
data_seed = 1234 if "--same-seed" in sys.argv else 1234 + rank     # each rank must draw different data
gen = torch.Generator(device="cuda").manual_seed(data_seed)
B, S, ACCUM, STEPS = 8, 1024, 4, 10

for step in range(STEPS + 3):
    if step == 3:                                      # 3 warm-up steps, then start the clock
        torch.cuda.synchronize()
        t = time.perf_counter()
    for i in range(ACCUM):
        x = torch.randint(0, 50257, (B, S), device="cuda", generator=gen)
        last = i == ACCUM - 1
        # no_sync skips the gradient all-reduce; only the last micro-batch communicates
        ctx = contextlib.nullcontext() if last or "--sync-every-micro" in sys.argv else model.no_sync()
        with ctx:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x, x) / ACCUM
            scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
tps = STEPS * ACCUM * B * S * world / (time.perf_counter() - t)
losses = [None] * world
dist.all_gather_object(losses, round(loss.item() * ACCUM, 6))
if rank == 0:
    print(f"GPUs {world}  {' '.join(sys.argv[1:]) or 'default'}  {tps:,.0f} tokens/s  last loss per rank {losses}")
dist.destroy_process_group()

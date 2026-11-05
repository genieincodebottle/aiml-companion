# bench.py - sustained DDP training throughput on real FineWeb-Edu tokens, fp16 autocast + GradScaler.
# Launch: torchrun --nproc_per_node=2 bench.py --size 124m --micro 12 [--compile] [--ckpt] [--resume-check]
import argparse, contextlib, datetime, json, os, random, time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, CONFIGS, flops_per_token

p = argparse.ArgumentParser()
p.add_argument("--size", default="124m")
p.add_argument("--micro", type=int, default=8)
p.add_argument("--tokens-per-step", type=int, default=524288)
p.add_argument("--accum", type=int, default=0)          # 0 = derive from tokens-per-step
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--compile", action="store_true")
p.add_argument("--ckpt", action="store_true")
p.add_argument("--resume-check", action="store_true")
p.add_argument("--data", default="/kaggle/working/data/train.bin")
p.add_argument("--tag", default="")
a = p.parse_args()

dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
cfg = CONFIGS[a.size]
T = cfg["ctx"]
accum = a.accum or max(1, round(a.tokens_per_step / (a.micro * T * world)))
tokens_per_step = accum * a.micro * T * world


class Loader:
    """Contiguous windows; rank r reads the r-th slice of each global micro-batch. pos is the resumable state."""
    def __init__(self, path, b):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.b, self.pos = b, 0

    def next(self):
        n = self.b * T
        if self.pos + world * n + 1 > len(self.data):
            self.pos = 0
        s = self.pos + rank * n
        w = torch.from_numpy(self.data[s:s + n + 1].astype(np.int64))
        self.pos += world * n
        x, y = w[:-1].view(self.b, T), w[1:].view(self.b, T)
        return x.pin_memory().cuda(non_blocking=True), y.pin_memory().cuda(non_blocking=True)


torch.manual_seed(0)
raw = GPT(**cfg, ckpt=a.ckpt).cuda()
N = sum(q.numel() for q in raw.parameters())
model = DDP(torch.compile(raw) if a.compile else raw, device_ids=[rank])
opt = torch.optim.AdamW(raw.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
scaler = torch.amp.GradScaler("cuda")
loader = Loader(a.data, a.micro)
step = 0


def train_step():
    global step
    lr = 6e-4 * min(1.0, (step + 1) / 10)
    for g in opt.param_groups:
        g["lr"] = lr
    tot = torch.zeros((), device="cuda")
    for i in range(accum):
        x, y = loader.next()
        ctx = contextlib.nullcontext() if i == accum - 1 else model.no_sync()
        with ctx:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x, y) / accum
            scaler.scale(loss).backward()
        tot += loss.detach()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce(tot, op=dist.ReduceOp.AVG)
    step += 1
    return tot.item(), gn.item()


rec = {"tag": a.tag, "size": a.size, "params": N, "micro": a.micro, "accum": accum, "world": world,
       "tokensPerStep": tokens_per_step, "compile": a.compile, "ckpt": a.ckpt, "steps": []}
try:
    t_first = time.perf_counter()
    for _ in range(a.warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, gn = train_step()
        torch.cuda.synchronize()
        rec["steps"].append({"step": step, "sec": round(time.perf_counter() - t0, 2), "loss": round(loss, 4), "warmup": True})
    rec["warmupSec"] = round(time.perf_counter() - t_first, 1)
    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    for _ in range(a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, gn = train_step()
        torch.cuda.synchronize()
        rec["steps"].append({"step": step, "sec": round(time.perf_counter() - t0, 2), "loss": round(loss, 4),
                             "gradNorm": round(gn, 3), "scale": scaler.get_scale()})
    dt = time.perf_counter() - t_all
    tps = a.steps * tokens_per_step / dt
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9], device="cuda")
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    fpt = flops_per_token(cfg, N)
    rec.update({"tokPerSec": round(tps), "secPerStep": round(dt / a.steps, 2), "peakAllocGB": round(peak[0].item(), 2),
                "peakReservedGB": round(peak[1].item(), 2), "flopsPerToken": fpt,
                "achievedTFLOPS": round(tps * fpt / 1e12, 2),
                "mfuVsSpec65": round(tps * fpt / (world * 65e12), 4),
                "mfuVsMeasured22p1": round(tps * fpt / (world * 22.1e12), 4)})
except torch.cuda.OutOfMemoryError:
    rec["oom"] = True
    rec["allocAtOomGB"] = round(torch.cuda.memory_allocated() / 1e9, 2)

if a.resume_check and not rec.get("oom"):
    # Exact resume: save full state, take 2 steps, reload into the live objects, take the same 2 steps again.
    path = "/tmp/capstone_ckpt.pt"
    t0 = time.perf_counter()
    # Each rank saves its own RNG streams; rank 0 also saves the replicated model, optimizer, scaler and loader.
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state()}
    torch.save(rng, f"/tmp/capstone_rng_{rank}.pt")
    if rank == 0:
        state = {"model": raw.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "loader_pos": loader.pos, "step": step}
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
    dist.barrier()
    save_sec = time.perf_counter() - t0
    cont = [train_step()[0] for _ in range(2)]
    t0 = time.perf_counter()
    # map_location="cpu": a CUDA map_location also moves the saved CPU RNG ByteTensor to the GPU, and
    # torch.set_rng_state then raises "RNG state must be a torch.ByteTensor" (measured, run llt-capstone-gpt2-measure2).
    st = torch.load(path, map_location="cpu", weights_only=False)
    raw.load_state_dict(st["model"])
    opt.load_state_dict(st["optimizer"])            # moves the AdamW state to the parameters' device
    scaler.load_state_dict(st["scaler"])
    loader.pos, step = st["loader_pos"], st["step"]
    r = torch.load(f"/tmp/capstone_rng_{rank}.pt", map_location="cpu", weights_only=False)
    random.setstate(r["python"])
    np.random.set_state(r["numpy"])
    torch.set_rng_state(r["torch"])
    torch.cuda.set_rng_state(r["cuda"])
    dist.barrier()
    load_sec = time.perf_counter() - t0
    again = [train_step()[0] for _ in range(2)]
    rec["resume"] = {"fileGB": round(os.path.getsize(path) / 1e9, 3), "saveSec": round(save_sec, 1),
                     "loadSec": round(load_sec, 1), "continued": cont, "reloaded": again,
                     "maxAbsDiff": max(abs(u - v) for u, v in zip(cont, again))}

if rank == 0:
    print("RESULT " + json.dumps(rec))
dist.destroy_process_group()

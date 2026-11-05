# train.py - pretrain the course GPT on FineWeb-Edu tokens, with a WSD schedule, exact resume and Run Library logs.
#
# One script for both paths:
#   H100 reference run (1 GPU, bf16):   torchrun --nproc_per_node=1 train.py --data /workspace/data --out /workspace/run
#   Kaggle 2x T4 (fp16 + GradScaler):   torchrun --nproc_per_node=2 train.py --micro 8 ...
#   Resume after a stop:                 add --resume auto
#   Learner decay from our checkpoint:   --resume <ckpt dir> --stable-end 4529 --max-steps 4769 --out <new dir>
#
# Schedule (WSD): linear warmup to --lr over --warmup steps, constant until --stable-end, then linear to 0 at --max-steps.
# Defaults reproduce the reference run: 19,073 steps x 524,288 tokens = 10.0B tokens, decay over the last 10%.
import argparse, contextlib, datetime, glob, json, math, os, random, shutil, time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from gpt import GPT, CONFIGS, flops_per_token

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--run-id", default="lltCapstoneGPT2-ref-124m-10B")
p.add_argument("--size", default="124m", choices=sorted(CONFIGS))
p.add_argument("--micro", type=int, default=32)
p.add_argument("--tokens-per-step", type=int, default=524_288)
p.add_argument("--max-steps", type=int, default=19_073)
p.add_argument("--warmup", type=int, default=700)
p.add_argument("--stable-end", type=int, default=17_166)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--no-compile", action="store_true")
p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16"])
p.add_argument("--save-every", type=int, default=500)
p.add_argument("--keep-last", type=int, default=2)
p.add_argument("--milestones", default="906,1813,4292,4529,9060,17166,19073")   # stable-phase branch points
p.add_argument("--export-every", type=int, default=1000)
p.add_argument("--val-every", type=int, default=250)
p.add_argument("--val-tokens", type=int, default=4_194_304)     # first 4M tokens of val.bin
p.add_argument("--sample-every", type=int, default=1000)
p.add_argument("--resume", default="")    # "auto", a checkpoint dir, or a state.pt file (e.g. downloaded from the Hub)
p.add_argument("--reset-loader-shard", type=int, default=-1)   # branch runs: start reading at this shard (seed = shard)
p.add_argument("--stop-after-sec", type=float, default=0)       # stop cleanly with a checkpoint (12 h sessions)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

# ---- process group, device, precision ------------------------------------------------------------------------
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
cap = torch.cuda.get_device_capability(rank)
# Pick by capability, not torch.cuda.is_bf16_supported(): that returns True on a T4, which has no bf16 tensor cores.
dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype, torch.bfloat16 if cap >= (8, 0) else torch.float16)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
master = rank == 0
cfg = CONFIGS[a.size]
T = cfg["ctx"]
accum = a.tokens_per_step // (a.micro * T * world)
assert accum * a.micro * T * world == a.tokens_per_step, "tokens-per-step must equal micro x 1024 x GPUs x accum"
milestones = {int(s) for s in a.milestones.split(",") if s}
ckpt_dir, w_dir = os.path.join(a.out, "ckpt"), os.path.join(a.out, "weights")
if master:
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)


# ---- data ----------------------------------------------------------------------------------------------------
class Loader:
    """Reads contiguous windows across shards. Each micro-step takes world*b*T tokens; rank r reads slice r.
    The state (shard, pos, epoch) is saved in every checkpoint, so a resumed run reads exactly the next tokens."""

    def __init__(self, data_dir, b):
        self.index = json.load(open(os.path.join(data_dir, "index.json")))
        self.shards = [np.memmap(os.path.join(data_dir, s["file"]), dtype=np.uint16, mode="r") for s in self.index["train"]]
        self.starts = [s["start"] for s in self.index["train"]]
        self.b, self.shard, self.pos, self.epoch = b, 0, self.starts[0], 0

    def state_dict(self):
        return {"shard": self.shard, "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, s):
        self.shard, self.pos, self.epoch = s["shard"], s["pos"], s["epoch"]

    def next(self):
        n = self.b * T
        while self.pos + world * n + 1 > len(self.shards[self.shard]):
            self.shard += 1
            if self.shard == len(self.shards):
                self.shard, self.epoch = 0, self.epoch + 1
            self.pos = self.starts[self.shard]
        s = self.pos + rank * n
        w = torch.from_numpy(self.shards[self.shard][s:s + n + 1].astype(np.int64))
        self.pos += world * n
        x, y = w[:-1].view(self.b, T), w[1:].view(self.b, T)
        return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)


loader = Loader(a.data, a.micro)
val_data = np.memmap(os.path.join(a.data, loader.index["val"]["file"]), dtype=np.uint16, mode="r")

# ---- model and optimizer -------------------------------------------------------------------------------------
torch.manual_seed(a.seed)
raw = GPT(**cfg).to(dev)
N = sum(q.numel() for q in raw.parameters())
decay = [q for q in raw.parameters() if q.dim() >= 2]       # matrices and the tied embedding get weight decay
no_decay = [q for q in raw.parameters() if q.dim() < 2]     # RMSNorm gains do not
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)   # bf16 needs no loss scaling
model = DDP(raw if a.no_compile else torch.compile(raw), device_ids=[rank])


def lr_at(step):                         # step counts completed optimizer steps; this is the LR for step+1
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if step < a.stable_end:
        return a.lr
    return a.lr * max(0.0, (a.max_steps - step) / max(1, a.max_steps - a.stable_end))


# ---- checkpoints -----------------------------------------------------------------------------------------------
def save_ckpt(step, run):
    final = os.path.join(ckpt_dir, f"step_{step:06d}")
    tmp = final + ".tmp"
    if master:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
    dist.barrier()
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(dev)}, os.path.join(tmp, f"rng_{rank}.pt"))
    if master:
        torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "loader": loader.state_dict(), "step": step, "args": vars(a), "config": cfg, "world": world,
                    "micro": a.micro, "dtype": str(dtype), "metrics": run["metrics"], "samples": run["samples"]},
                   os.path.join(tmp, "state.pt"))
    dist.barrier()
    if master:
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)                # a session that dies mid-save leaves the previous checkpoint intact
        done = sorted(d for d in glob.glob(os.path.join(ckpt_dir, "step_*")) if not d.endswith(".tmp"))
        rolling = [d for d in done if int(d[-6:]) not in milestones]
        for d in rolling[:-a.keep_last]:
            shutil.rmtree(d, ignore_errors=True)
    dist.barrier()


def export_weights(step):
    if master:
        sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in raw.state_dict().items()}
        torch.save({"config": cfg, "step": step, "tokens": step * a.tokens_per_step, "state_dict": sd},
                   os.path.join(w_dir, f"weights_step_{step:06d}.pt"))


def find_resume():
    if a.resume != "auto":
        return a.resume
    done = sorted(d for d in glob.glob(os.path.join(ckpt_dir, "step_*")) if not d.endswith(".tmp"))
    return done[-1] if done else ""


# ---- eval and samples --------------------------------------------------------------------------------------------
@torch.no_grad()
def val_loss():
    raw.eval()
    n = a.micro * T
    chunks = (min(a.val_tokens, len(val_data) - 1)) // n
    tot, cnt = torch.zeros((), device=dev), 0
    for c in range(rank, chunks, world):
        w = torch.from_numpy(val_data[c * n:c * n + n + 1].astype(np.int64)).to(dev)
        with torch.autocast("cuda", dtype=dtype):
            tot += raw(w[:-1].view(a.micro, T), w[1:].view(a.micro, T))
        cnt += 1
    stats = torch.tensor([tot.item(), cnt], device=dev, dtype=torch.float64)
    dist.all_reduce(stats)
    raw.train()
    return (stats[0] / stats[1]).item(), int(stats[1].item()) * n


PROMPTS = ["The water cycle is", "In mathematics, a prime number", "The French Revolution began"]


@torch.no_grad()
def samples(step):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    raw.eval()
    out = []
    g = torch.Generator(device=dev).manual_seed(1234)
    for prompt in PROMPTS:
        ids = torch.tensor([enc.encode_ordinary(prompt)], device=dev)
        for _ in range(48):
            with torch.autocast("cuda", dtype=dtype):
                logits = raw(ids)[:, -1, :].float()
            v, ix = torch.topk(logits, 40)
            nxt = ix.gather(1, torch.multinomial(torch.softmax(v / 0.8, -1), 1, generator=g))
            ids = torch.cat([ids, nxt], 1)
        out.append({"step": step, "prompt": prompt, "completion": enc.decode(ids[0].tolist())[len(prompt):]})
    raw.train()
    return out


def write_run(run, step, t_train):
    if not master:
        return
    run.update({"wallClockSec": round(t_train, 1), "lastStep": step, "date": time.strftime("%Y-%m-%d")})
    with open(os.path.join(a.out, a.run_id + ".json.tmp"), "w") as f:
        json.dump(run, f, indent=1)
    os.replace(os.path.join(a.out, a.run_id + ".json.tmp"), os.path.join(a.out, a.run_id + ".json"))


# ---- resume ------------------------------------------------------------------------------------------------------
step, prior_sec = 0, 0.0
run = {"runId": a.run_id, "module": "lltCapstoneGPT2", "tier": "T4" if cap >= (8, 0) else "T2", "label": "baseline",
       "hardware": {"gpu": torch.cuda.get_device_name(rank), "count": world, "capability": list(cap),
                    "vramGB": round(torch.cuda.get_device_properties(rank).total_memory / 1e9, 1)},
       "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "dtype": str(dtype), "compile": not a.no_compile},
       "notebookCommit": None, "config": {**vars(a), "params": N, "accum": accum, "model": cfg}, "seed": a.seed,
       "metrics": [], "evals": {}, "samples": [], "resumes": [], "costUSD": None, "priceCheckedOn": None}
src = find_resume()
if src:
    state_path = src if src.endswith(".pt") else os.path.join(src, "state.pt")
    # state.pt holds only tensors and plain Python types, so it loads with weights_only=True (safe for downloads)
    st = torch.load(state_path, map_location="cpu", weights_only=True)
    raw.load_state_dict(st["model"])
    opt.load_state_dict(st["optimizer"])
    for g in opt.param_groups:            # the schedule comes from this run's arguments, not the saved ones
        g["weight_decay"] = a.wd if g["weight_decay"] else 0.0
    if scaler.is_enabled() and st["scaler"]:
        scaler.load_state_dict(st["scaler"])   # a bf16 (H100) checkpoint has an empty scaler state; start fresh
    loader.load_state_dict(st["loader"])
    step = st["step"]
    run["metrics"], run["samples"] = st["metrics"], st["samples"]
    rng_file = os.path.join(os.path.dirname(state_path), f"rng_{rank}.pt")
    same_layout = st["world"] == world and st["micro"] == a.micro and st["dtype"] == str(dtype)
    if os.path.exists(rng_file) and same_layout:
        # RNG files hold NumPy state, so weights_only=False; they are only ever our own local files, never downloaded.
        # map_location="cpu": a CUDA map_location breaks set_rng_state.
        r = torch.load(rng_file, map_location="cpu", weights_only=False)
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        torch.cuda.set_rng_state(r["cuda"], dev)
    if a.reset_loader_shard >= 0:
        loader.load_state_dict({"shard": a.reset_loader_shard, "pos": loader.starts[a.reset_loader_shard], "epoch": 0})
    prior_sec = st["metrics"][-1]["wallSec"] if st["metrics"] else 0.0
    run["resumes"].append({"from": src, "step": step, "sameLayout": same_layout, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if master:
        print(f"resumed from {src} at step {step} (same layout: {same_layout})", flush=True)

# ---- train -------------------------------------------------------------------------------------------------------
fpt = flops_per_token(cfg, N)
t_start = time.time()
if master:
    print(f"{a.size} params {N:,} | {torch.cuda.get_device_name(rank)} x{world} | {dtype} | micro {a.micro} x accum {accum}"
          f" | {a.tokens_per_step:,} tokens/step | steps {step} -> {a.max_steps}", flush=True)
while step < a.max_steps:
    torch.cuda.synchronize()
    t0 = time.time()
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    tot = torch.zeros((), device=dev)
    for i in range(accum):
        x, y = loader.next()
        ctx = contextlib.nullcontext() if i == accum - 1 else model.no_sync()
        with ctx:
            with torch.autocast("cuda", dtype=dtype):
                loss = model(x, y) / accum
            scaler.scale(loss).backward()
        tot += loss.detach()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    dist.all_reduce(tot, op=dist.ReduceOp.AVG)
    step += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    m = {"step": step, "tokens": step * a.tokens_per_step, "trainLoss": round(tot.item(), 5), "lr": lr,
         "gradNorm": round(gn.item(), 4), "tokPerSec": round(a.tokens_per_step / dt), "stepSec": round(dt, 3),
         "mfu": round(a.tokens_per_step / dt * fpt / (world * (989e12 if cap >= (9, 0) else 312e12 if cap >= (8, 0) else 65e12)), 4),
         "memGB": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2), "loaderShard": loader.shard,
         "epoch": loader.epoch, "wallSec": round(prior_sec + time.time() - t_start, 1)}
    if scaler.is_enabled():
        m["lossScale"] = scaler.get_scale()
    if step % a.val_every == 0 or step == a.max_steps:
        m["valLoss"], m["valTokens"] = val_loss()
        m["valLoss"] = round(m["valLoss"], 5)
    if master and (step % a.sample_every == 0 or step == a.max_steps):
        run["samples"] += samples(step)
    run["metrics"].append(m)
    if master and (step % 10 == 0 or "valLoss" in m):
        print(json.dumps(m), flush=True)
    if step % a.export_every == 0 or step in milestones or step == a.max_steps:
        export_weights(step)
    out_of_time = a.stop_after_sec and time.time() - t_start > a.stop_after_sec
    stop = torch.tensor(int(bool(out_of_time)), device=dev)
    dist.all_reduce(stop, op=dist.ReduceOp.MAX)          # every rank must agree to stop
    if step % a.save_every == 0 or step in milestones or step == a.max_steps or stop.item():
        save_ckpt(step, run)
        write_run(run, step, prior_sec + time.time() - t_start)
    if stop.item():
        if master:
            print(f"stopping at step {step} after {time.time() - t_start:.0f} s; rerun with --resume auto", flush=True)
        break

write_run(run, step, prior_sec + time.time() - t_start)
dist.destroy_process_group()

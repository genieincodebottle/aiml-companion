# fsdp_ckpt.py - launch with: torchrun --nproc_per_node=2 fsdp_ckpt.py
import gc, os, time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from fsdp import build, train, CFG
from gpt import GPT

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
PATH = "/tmp/ckpt_fsdp"

model, opt = build()
scaler = torch.amp.GradScaler("cuda")
train(model, opt, scaler, 3, torch.Generator(device="cuda").manual_seed(100 + rank))

# Wrong. Rank 0 alone saves its state_dict. Under FSDP every tensor in it holds only rank 0's shard.
if rank == 0:
    torch.save(model.state_dict(), "/tmp/rank0_only.pt")

# Right. Every rank writes its own shards, model and optimizer, plus the loss scaler.
t = time.perf_counter()
msd, osd = get_state_dict(model, opt)
dcp.save({"model": msd, "optim": osd, "scaler": scaler.state_dict()}, checkpoint_id=PATH)
dist.barrier()
save_s = time.perf_counter() - t
loss_a = train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))
del model, opt, msd, osd
gc.collect()
torch.cuda.empty_cache()

# A fresh model and optimizer load the checkpoint, then take the same 2 steps on the same batches.
model, opt = build()
t = time.perf_counter()
msd, osd = get_state_dict(model, opt)
state = {"model": msd, "optim": osd, "scaler": scaler.state_dict()}
dcp.load(state, checkpoint_id=PATH)
set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
scaler.load_state_dict(state["scaler"])
load_s = time.perf_counter() - t
loss_b = train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))

if rank == 0:
    files = {f: round(os.path.getsize(os.path.join(PATH, f)) / 1e9, 2) for f in sorted(os.listdir(PATH))}
    print("sharded checkpoint files (GB):", files)
    print(f"save {save_s:.1f} s  load {load_s:.1f} s  loss after 2 more steps: continued {loss_a:.6f}  reloaded {loss_b:.6f}")
    sd = torch.load("/tmp/rank0_only.pt", weights_only=False)
    kept = sum(v.to_local().numel() for v in sd.values())
    print(f"rank-0-only file {os.path.getsize('/tmp/rank0_only.pt') / 1e9:.2f} GB, holds {kept:,} values for a model of "
          f"{sum(v.numel() for v in sd.values()):,}")
    try:
        GPT(CFG).load_state_dict({k: v.to_local() for k, v in sd.items()})
    except RuntimeError as e:
        first = [line.strip() for line in str(e).splitlines() if "size mismatch" in line][:1]
        print("loading it into a full model:", first or str(e)[:300])
dist.destroy_process_group()

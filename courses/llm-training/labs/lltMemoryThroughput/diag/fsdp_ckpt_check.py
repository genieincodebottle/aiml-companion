# fsdp_ckpt_check.py - diagnostic, not on the page. Does reloading the same checkpoint twice give
# bit-identical losses? If yes, the 5e-5 gap between the continued and reloaded runs in fsdp_ckpt.py
# is a restore difference. If two reloads also differ, it is run-to-run GPU nondeterminism.
import gc, torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from fsdp import build, train

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
PATH = "/tmp/ckpt_check"
model, opt = build()
scaler = torch.amp.GradScaler("cuda")
train(model, opt, scaler, 3, torch.Generator(device="cuda").manual_seed(100 + rank))
msd, osd = get_state_dict(model, opt)
dcp.save({"model": msd, "optim": osd, "scaler": scaler.state_dict()}, checkpoint_id=PATH)
dist.barrier()
results = {"continued": train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))}
del model, opt, msd, osd
gc.collect(); torch.cuda.empty_cache()
for rep in ("reload1", "reload2"):
    model, opt = build()
    msd, osd = get_state_dict(model, opt)
    state = {"model": msd, "optim": osd, "scaler": scaler.state_dict()}
    dcp.load(state, checkpoint_id=PATH)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    scaler.load_state_dict(state["scaler"])
    results[rep] = train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))
    del model, opt, msd, osd, state
    gc.collect(); torch.cuda.empty_cache()
out = [None] * dist.get_world_size()
dist.all_gather_object(out, {k: f"{v:.6f}" for k, v in results.items()})
if rank == 0:
    print("losses after 2 steps per rank:", out)
dist.destroy_process_group()

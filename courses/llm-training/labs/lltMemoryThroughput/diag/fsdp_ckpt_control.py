# fsdp_ckpt_control.py - diagnostic, not on the page. Negative control for step 10's resume check.
# Save after 3 steps, then compare the loss after 2 more steps for (a) the continued run, (b) a full reload of
# model, optimizer and scaler, (c) a reload of the weights only, with a fresh AdamW. Also checksum every saved
# tensor shard before the save and after the full reload.
import gc, json, torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict
from fsdp import build, train

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
PATH = "/tmp/ckpt_control"


def checksums(model, opt):
    msd, osd = get_state_dict(model, opt)
    out = {}
    for k, v in msd.items():
        out["model." + k] = v.to_local().double().sum().item()
    for fqn, st in osd["state"].items():
        for name, v in st.items():
            if torch.is_tensor(v):
                t = v.to_local() if hasattr(v, "to_local") else v
                out[f"optim.{fqn}.{name}"] = t.double().sum().item()
    return out


def fresh():
    gc.collect()
    torch.cuda.empty_cache()
    return build()


model, opt = build()
scaler = torch.amp.GradScaler("cuda")
train(model, opt, scaler, 3, torch.Generator(device="cuda").manual_seed(100 + rank))
saved = checksums(model, opt)
msd, osd = get_state_dict(model, opt)
dcp.save({"model": msd, "optim": osd, "scaler": scaler.state_dict()}, checkpoint_id=PATH)
dist.barrier()
scaler_state = scaler.state_dict()
res = {"continued": train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))}
del model, opt, msd, osd

# (b) full reload
model, opt = fresh()
msd, osd = get_state_dict(model, opt)
state = {"model": msd, "optim": osd, "scaler": scaler_state}
dcp.load(state, checkpoint_id=PATH)
set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
scaler = torch.amp.GradScaler("cuda")
scaler.load_state_dict(state["scaler"])
loaded = checksums(model, opt)
mismatch = sum(1 for k in saved if saved[k] != loaded.get(k))
res["fullReload"] = train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))
del model, opt, msd, osd, state

# (c) weights only, fresh AdamW, same scaler state
model, opt = fresh()
msd = get_model_state_dict(model)
state = {"model": msd}
dcp.load(state, checkpoint_id=PATH)
set_model_state_dict(model, state["model"])
scaler = torch.amp.GradScaler("cuda")
scaler.load_state_dict(scaler_state)
res["weightsOnlyFreshAdam"] = train(model, opt, scaler, 2, torch.Generator(device="cuda").manual_seed(7 + rank))

out = [None] * dist.get_world_size()
dist.all_gather_object(out, {"losses": {k: f"{v:.6f}" for k, v in res.items()}, "tensorsChecked": len(saved),
                             "checksumMismatches": mismatch})
if rank == 0:
    print("RESULT " + json.dumps(out))
dist.destroy_process_group()

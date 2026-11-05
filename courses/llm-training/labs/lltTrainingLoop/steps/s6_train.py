import json
import time
from contextlib import nullcontext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype():
    # T4 is compute capability 7.5: fp16 tensor cores, no bf16 ones, although
    # torch.cuda.is_bf16_supported() returns True there. Decide by capability.
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16


@torch.no_grad()
def evaluate(model, cfg, dtype):
    # Mean loss over every window of the validation file, in a fixed order.
    model.eval()
    val = np.memmap("data/val.bin", dtype=np.uint8, mode="r")
    T = cfg["block"]
    starts = np.arange(0, len(val) - T - 1, T)
    total, count = 0.0, 0
    for i in range(0, len(starts), 64):
        s = starts[i:i + 64]
        x = torch.from_numpy(np.stack([val[j:j + T] for j in s]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[j + 1:j + 1 + T] for j in s]).astype(np.int64)).to(device)
        with autocast(dtype):
            logits = model(x)
        total += F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count


def autocast(dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def train(cfg, log_path=None):
    torch.manual_seed(cfg["seed"])
    dtype = pick_dtype()
    model = GPT(len(chars), cfg["block"], cfg["n_layer"], cfg["n_head"], cfg["d"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=cfg["lr"], betas=(0.9, 0.99))
    sched = make_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))
    sampler = Sampler("data/train.bin", cfg["block"], cfg["micro_batch"], cfg["seed"])
    step = 0
    if cfg.get("resume"):
        step = load_checkpoint(cfg["ckpt"], model, opt, sched, scaler, sampler, skip=cfg.get("skip", ()))
    elif log_path:
        open(log_path, "w").close()
    N, history, t0 = cfg["accum"], [], time.time()
    while step < cfg.get("stop_at", cfg["steps"]):
        opt.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(N):
            x, y = (t.to(device) for t in sampler.next_batch())
            with autocast(dtype):
                loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
            scaler.scale(loss / N if cfg.get("divide_loss", True) else loss).backward()
            micro_losses.append(loss.item())
            if cfg.get("sched_per_micro"):
                sched.step()                     # broken on purpose in step 10
        scaler.unscale_(opt)                     # clip the true gradients, not the scaled ones
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]).item()
        lr = opt.param_groups[0]["lr"]
        scaler.step(opt)                         # skips the update if fp16 gradients overflowed
        scaler.update()
        if not cfg.get("sched_per_micro"):
            sched.step()
        step += 1
        row = {"step": step, "trainLoss": sum(micro_losses) / N, "gradNorm": norm, "lr": lr}
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            row["valLoss"] = evaluate(model, cfg, dtype)
            row["elapsedSec"] = round(time.time() - t0, 1)
        history.append(row)
        if log_path:
            with open(log_path, "a") as f:
                print(json.dumps(row), file=f)
        if cfg.get("ckpt") and step % cfg.get("ckpt_every", 10**9) == 0:
            save_checkpoint(cfg["ckpt"], model, opt, sched, scaler, sampler, step, cfg)
    return model, history

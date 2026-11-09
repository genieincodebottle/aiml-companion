import torch.nn.functional as F
from peft import PeftModel
RUN_DPO = False          # True trains DPO on your T4; False downloads our adapter from the Hub

def reference_logps(model, pairs, rows=8):
    # one pass of the frozen SFT model over every pair, so training never holds a reference model
    out = torch.zeros(len(pairs), 2)
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i]["c"]) - len(pairs[i]["r"]))
    model.eval()
    for k in range(0, len(order), rows // 2):
        idx = order[k:k + rows // 2]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lp = seq_logps(model, [pairs[i]["c"] for i in idx] + [pairs[i]["r"] for i in idx], [len(pairs[i]["p"]) for i in idx] * 2)
        out[idx, 0], out[idx, 1] = lp[:len(idx)].float().cpu(), lp[len(idx):].float().cpu()
    return out

t0 = time.time()
ref_dpo, ref_held = reference_logps(sft, dpo_pairs), reference_logps(sft, held)
runs["refSec"] = round(time.time() - t0, 1)
print("reference log-probs for", len(dpo_pairs) + len(held), "pairs in", runs["refSec"], "s | mean chosen", round(ref_dpo[:, 0].mean().item(), 1),
      "rejected", round(ref_dpo[:, 1].mean().item(), 1))
del sft
gc.collect()
torch.cuda.empty_cache()

def pref_loss(kind, pc, pr, rc, rr, nc, nr, beta, gamma=1.0, kl=None):
    # pc, pr: policy reply log-probs; rc, rr: reference; nc, nr: reply lengths in tokens. One loss per pair.
    if kind == "dpo":
        return -F.logsigmoid(beta * ((pc - rc) - (pr - rr)))
    if kind == "ipo":            # TRL's form: log-ratios averaged per token, squared distance to 1 / (2 beta)
        return ((pc - rc) / nc - (pr - rr) / nr - 1 / (2 * beta)) ** 2
    if kind == "simpo":          # no reference: per-token log-probs and a target margin gamma
        return -F.logsigmoid(beta * (pc / nc - pr / nr) - gamma)
    if kind == "orpo":           # no reference: NLL of the chosen reply plus beta times an odds-ratio loss
        ac, ar = pc / nc, pr / nr
        log_odds = (ac - ar) - (torch.log(-torch.expm1(ac)) - torch.log(-torch.expm1(ar)))
        return -ac - beta * F.logsigmoid(log_odds)
    if kind == "kto":            # chosen and rejected scored on their own against a KL reference point
        return 0.5 * ((1 - torch.sigmoid(beta * ((pc - rc) - kl))) + (1 - torch.sigmoid(beta * (kl - (pr - rr)))))
    raise ValueError(kind)

def kto_kl(model, part):
    # KL estimate from mismatched rows: prompt i with the chosen reply of pair i + 1, policy minus reference, no gradient
    rows = [x["p"] + part[(i + 1) % len(part)]["c"][len(part[(i + 1) % len(part)]["p"]):] for i, x in enumerate(part)]
    plens = [len(x["p"]) for x in part]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pol = seq_logps(model, rows, plens)
        with model.disable_adapter():
            ref = seq_logps(model, rows, plens)
    return (pol - ref).mean().clamp(min=0)

def held_stats(model, step):
    # policy minus reference log-prob of every held-out reply, summarised
    lp = torch.zeros(len(held), 2)
    model.eval()
    for k in range(0, len(held), 4):
        part = held[k:k + 4]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            x = seq_logps(model, [h["c"] for h in part] + [h["r"] for h in part], [len(h["p"]) for h in part] * 2).float().cpu()
        lp[k:k + len(part), 0], lp[k:k + len(part), 1] = x[:len(part)], x[len(part):]
    model.train()
    dc, dr = (lp[:, 0] - ref_held[:, 0]).numpy(), (lp[:, 1] - ref_held[:, 1]).numpy()
    return {"step": step, "chosen": round(float(dc.mean()), 2), "rejected": round(float(dr.mean()), 2),
            "margin": round(float((dc - dr).mean()), 2), "accuracy": round(float((dc > dr).mean()), 3),
            "chosenDown": round(float((dc < 0).mean()), 3), "bothDown": round(float(((dc < 0) & (dr < 0)).mean()), 3), "dc": dc, "dr": dr}

def train_pref(pairs, ref, kind="dpo", beta=0.1, lr=5e-5, seed=0, size=8, micro=2, gamma=1.0, every=25, epochs=1):
    torch.manual_seed(seed)
    model = lora_on(load_sft())
    steps = [b for e in range(epochs) for b in plan(len(pairs), size, seed + 1000 * e, [len(x["c"]) + len(x["r"]) for x in pairs])]
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, len(steps), max(1, len(steps) // 10)))
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    t0, log, track, cost = time.time(), [], [], {"evalSec": 0.0, "trainPeak": 0, "evalPeak": 0}

    def check(step):
        # held-out log-probs, timed and kept out of the training memory peak
        if step > 0:                             # the peak since the last check is a training peak
            cost["trainPeak"] = max(cost["trainPeak"], torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        te = time.time()
        track.append(held_stats(model, step))
        cost["evalSec"] += time.time() - te
        cost["evalPeak"] = max(cost["evalPeak"], torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    torch.cuda.reset_peak_memory_stats()        # forget the reference pass and anything before this run
    check(0)
    model.train()
    for step, idx in enumerate(steps):
        opt.zero_grad(set_to_none=True)
        row = {"loss": 0.0, "dc": [], "dr": []}
        for k in range(0, len(idx), micro):
            j = idx[k:k + micro]
            part = [pairs[i] for i in j]
            with torch.autocast("cuda", dtype=torch.float16):
                lp = seq_logps(model, [x["c"] for x in part] + [x["r"] for x in part], [len(x["p"]) for x in part] * 2)
            pc, pr = lp[:len(part)].float(), lp[len(part):].float()
            rc, rr = ref[j, 0].to(GPU), ref[j, 1].to(GPU)
            nc = torch.tensor([len(x["c"]) - len(x["p"]) for x in part], device=GPU)
            nr = torch.tensor([len(x["r"]) - len(x["p"]) for x in part], device=GPU)
            kl = kto_kl(model, part) if kind == "kto" else None
            loss = pref_loss(kind, pc, pr, rc, rr, nc, nr, beta, gamma, kl).sum() / len(idx)
            scaler.scale(loss).backward()
            row["loss"] += loss.item()
            row["dc"] += (pc - rc).tolist()
            row["dr"] += (pr - rr).tolist()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        log.append({"loss": round(row["loss"], 4), "chosen": round(float(np.mean(row["dc"])), 3), "rejected": round(float(np.mean(row["dr"])), 3)})
        if (step + 1) % every == 0:
            check(step + 1)
            h = track[-1]
            print(kind, "beta", beta, "| step", step + 1, "of", len(steps), "| train loss", round(float(np.mean([x["loss"] for x in log[-every:]])), 4),
                  "| held-out chosen", h["chosen"], "rejected", h["rejected"], "accuracy", h["accuracy"], "| seconds", round(time.time() - t0))
    if len(steps) % every:                       # the last step was not checked yet
        check(len(steps))
    info = {"kind": kind, "beta": beta, "lr": lr, "seed": seed, "pairs": len(pairs), "epochs": epochs, "steps": len(steps), "sec": round(time.time() - t0, 1),
            "trainSec": round(time.time() - t0 - cost["evalSec"], 1), "trainPeakGB": round(cost["trainPeak"] / 1e9, 2),
            "evalPeakGB": round(cost["evalPeak"] / 1e9, 2), "log": log,
            "held": [{k: v for k, v in h.items() if k not in ("dc", "dr")} for h in track]}
    return model, info, track[-1]

if RUN_DPO:
    dpo, info, final = train_pref(dpo_pairs, ref_dpo, "dpo", beta=0.1)
    dpo.save_pretrained("dpo-b0.1-s0")
    runs["dpo-b0.1-s0"] = info
    print("training peak GB", info["trainPeakGB"], "| held-out check peak GB", info["evalPeakGB"], "| seconds", info["sec"],
          "of which held-out checks", round(info["sec"] - info["trainSec"], 1))
else:
    path = snapshot_download("aimlcompanion/llm-training-lab", allow_patterns="lltDPO/dpo-b0.1-s0/*")
    dpo = PeftModel.from_pretrained(load_sft(), os.path.join(path, "lltDPO", "dpo-b0.1-s0"))
    final = held_stats(dpo, 125)
print({k: v for k, v in final.items() if k not in ("dc", "dr")})

"""One lltLoRA run in its own process. Usage: python lora_worker.py <job.json> <result.json>

job keys: tag, model, mode (full|lora|eval), lr, opt (adamw8bit|adamw32), r, alpha, targets (all|attn|mlp),
rslora, b_init (zero|random), n_train, steps (default n_train // batch), batch, budget, seed, pad_to,
eval_every, n_heldout, n_wiki, n_acc, spectra, truncate (list of ranks), save_adapter, pool (list of module
names whose pooled singular vectors are saved for the overview).
"""
import gc, json, os, sys, time, traceback, platform, io, logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
job = json.load(open(sys.argv[1])); outp = sys.argv[2]
R = {"tag": job["tag"], "job": job}
def save():
    json.dump(R, open(outp, "w"), indent=1)
import torch, transformers, peft, datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lora_lab as L
CUDA = os.environ.get("LORA_DEV", "cuda") == "cuda"
def gb(f):
    return round(f() / 1e9, 2) if CUDA else 0.0
from transformers import AutoModelForCausalLM, AutoTokenizer
logbuf = io.StringIO(); h = logging.StreamHandler(logbuf); h.setLevel(logging.WARNING)
logging.getLogger("transformers").addHandler(h)

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def linear_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head")]


def evals(model, tok, held, wiki, acc_pairs, dev, tag=""):
    t = time.time()
    e = {"heldoutLoss": round(L.mean_loss(model, held, tok.pad_token_id, dev), 5),
         "wikiLoss": round(L.mean_loss(model, wiki, tok.pad_token_id, dev), 5)}
    if acc_pairs:
        s, l, outs = L.accuracy(model, tok, acc_pairs, dev)
        e.update(accStrict=round(s, 4), accLenient=round(l, 4), samples=outs[:3])
    e["evalSec"] = round(time.time() - t, 1)
    print("eval", tag, {k: v for k, v in e.items() if k != "samples"}, flush=True)
    return e


try:
    dev = os.environ.get("LORA_DEV", "cuda")
    R["env"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": transformers.__version__, "peft": peft.__version__, "datasets": datasets.__version__,
                "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    try:
        import bitsandbytes; R["env"]["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        pass
    batch = job.get("batch", 16); n_train = job.get("n_train", 2000)
    steps = job.get("steps") or n_train // batch
    tok = AutoTokenizer.from_pretrained(job["model"])
    pairs = L.load_gsm8k("train")
    import random
    random.Random(1234).shuffle(pairs)                     # one fixed shuffle; every run draws its subset from it
    rows = L.encode(tok, pairs[:n_train])
    if job.get("pad_to"):
        P = job["pad_to"]
        rows = [(x + [tok.pad_token_id] * (P - len(x)), y + [-100] * (P - len(y))) for x, y in rows]
    test = L.load_gsm8k("test")
    held = L.encode(tok, test[:job.get("n_heldout", 300)])
    held_small = held[:100]
    wiki = L.text_rows(tok, job.get("n_wiki", 32))
    acc_pairs = test[:job.get("n_acc", 200)] if job.get("n_acc", 200) else []
    R["data"] = {"nTrain": len(rows), "trainAnswerTokens": sum(sum(1 for t in y if t != -100) for _, y in rows),
                 "trainTokens": sum(len(x) for x, _ in rows), "heldout": len(held), "wikiRows": len(wiki),
                 "accProblems": len(acc_pairs)}
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(job["model"], dtype=torch.float32, device_map=dev)
    R["loadSec"] = round(time.time() - t0, 1)
    R["totalParamsBase"] = sum(p.numel() for p in model.parameters())
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lin = linear_names(model)
    W0 = None
    if job["mode"] == "full" and (job.get("spectra") or job.get("truncate")):
        W0 = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    if job.get("eval_base"):
        R["evalBase"] = evals(model, tok, held, wiki, acc_pairs, dev, "base")
    if job["mode"] == "lora":
        from peft import LoraConfig, get_peft_model
        tgt = {"all": "all-linear", "attn": ATTN, "mlp": MLP}[job.get("targets", "all")]
        cfg = LoraConfig(r=job["r"], lora_alpha=job.get("alpha", 32), lora_dropout=job.get("dropout", 0.0),
                         target_modules=tgt, use_rslora=job.get("rslora", False),
                         init_lora_weights=job.get("b_init", "zero") == "zero", task_type="CAUSAL_LM")
        model.enable_input_require_grads()
        torch.manual_seed(job.get("seed", 0))
        model = get_peft_model(model, cfg)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    R["trainableParams"] = sum(p.numel() for p in params)
    R["totalParams"] = sum(p.numel() for p in model.parameters())
    if job["mode"] != "eval":
        if job.get("opt", "adamw32") == "adamw8bit":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=job["lr"], weight_decay=0.0)
        elif job.get("opt") == "adamw32-loop":
            # one tensor at a time: no multi-tensor temporaries, the same maths as the default AdamW
            opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=0.0, foreach=False)
        else:
            opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=0.0)
    R["memAfterLoadGB"] = gb(torch.cuda.memory_allocated)
    curve = []
    if job.get("eval_every"):
        curve.append({"step": 0, "heldoutLoss": round(L.mean_loss(model, held_small, tok.pad_token_id, dev), 5)})
        print("step 0", curve[-1], flush=True)
    save()
    hist = []
    if job["mode"] != "eval":
        if CUDA:
            torch.cuda.reset_peak_memory_stats()
        ee = job.get("eval_every") or 0
        def on_step(rec):
            rec["memGB"] = gb(torch.cuda.max_memory_allocated)
            if rec["step"] % 10 == 0 or rec["step"] <= 3:
                print(rec, flush=True)
            if ee and rec["step"] % ee == 0 and rec["step"] < steps:
                curve.append({"step": rec["step"], "heldoutLoss": round(L.mean_loss(model, held_small, tok.pad_token_id, dev), 5)})
                print("curve", curve[-1], flush=True)
        t1 = time.time()
        hist = L.train(model, rows, steps, opt, tok.pad_token_id, dev, batch=batch, budget=job.get("budget", 1024),
                       warmup=job.get("warmup", 10), seed=job.get("seed", 0), on_step=on_step)
        R["trainSec"] = round(time.time() - t1, 1)
        R["peakMemGB"] = gb(torch.cuda.max_memory_allocated)
        R["peakReservedGB"] = gb(torch.cuda.max_memory_reserved)
        secs = [h["sec"] for h in hist[3:]] or [h["sec"] for h in hist]
        R["secPerStepMedian"] = round(sorted(secs)[len(secs) // 2], 3)
        R["skippedSteps"] = sum(h["skipped"] for h in hist)
        R["nonFinite"] = sum(1 for h in hist if not (h["trainLoss"] == h["trainLoss"]) or abs(h["trainLoss"]) == float("inf"))
    R["metrics"] = hist
    R["curve"] = curve
    save()
    if job.get("eval_final", True):
        R["evalFinal"] = evals(model, tok, held, wiki, acc_pairs, dev, "final")
        if job.get("eval_every"):
            # same 100 examples as the earlier curve points; evalFinal uses all n_heldout
            curve.append({"step": steps, "heldoutLoss": round(L.mean_loss(model, held_small, tok.pad_token_id, dev), 5)})
    save()
    if job.get("save_adapter") and job["mode"] == "lora":
        model.save_pretrained(job["save_adapter"])
        R["adapterBytes"] = sum(os.path.getsize(os.path.join(job["save_adapter"], f)) for f in os.listdir(job["save_adapter"]))
    if job.get("dropout_eval_check"):
        # LoRA dropout left on at evaluation: score the same held-out set twice in train mode.
        model.train()
        vals = []
        for _ in range(2):
            tot, n = 0.0, 0
            with torch.no_grad():
                for group in L.micro_batches(held_small, 2048):
                    ids, mask, labels = L.collate(group, tok.pad_token_id, dev)
                    k = int((labels[:, 1:] != -100).sum())
                    with L.amp(dev):
                        tot += model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=k).loss.item() * k
                    n += k
            vals.append(round(tot / n, 5))
        R["dropoutEvalTrainMode"] = vals
        R["dropoutEvalEvalMode"] = round(L.mean_loss(model, held_small, tok.pad_token_id, dev), 5)
        save()
    if job["mode"] == "full" and W0 is not None:
        sd = dict(model.named_parameters())
        spec = {}; pooled = {}
        lin_set = set(lin)
        tops = {}
        for n in lin:
            W = sd[n + ".weight"].detach()
            d = W.float() - W0[n + ".weight"].to(dev)
            fro = float(d.norm())
            if job.get("truncate"):
                U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                K = min(256, S.numel())
                tops[n] = (U[:, :K].cpu(), S.cpu(), Vh[:K].cpu())
            else:
                S = torch.linalg.svdvals(d)
            s = S.cpu()
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            spec[n] = {"shape": list(d.shape), "fro": fro, "relChange": fro / float(W0[n + ".weight"].norm()),
                       "s": [float(f"{v:.5g}") for v in s.tolist()],
                       "energy": {str(r): round(float(e[min(r, len(e)) - 1]), 5) for r in (1, 2, 4, 8, 16, 32, 64, 128, 256)},
                       "r90": int((e < 0.9).sum()) + 1, "r50": int((e < 0.5).sum()) + 1}
            if n in job.get("pool", []):
                # Pool rows and columns into 64 blocks each. Pooling is linear, so the pooled rank-r
                # reconstruction equals the sum over the top r of (pooled u_i) s_i (pooled v_i).
                Uf, Sf, Vhf = torch.linalg.svd(d, full_matrices=False)
                m, k = d.shape
                pu = Uf[:, :64].reshape(64, m // 64, 64).mean(1) if m % 64 == 0 else None
                pv = Vhf[:64].T.reshape(64, k // 64, 64).mean(1) if k % 64 == 0 else None
                pooled[n] = {"shape": [m, k], "s": [float(f"{v:.6g}") for v in Sf[:64].tolist()],
                             "U64": [[round(float(v), 6) for v in row] for row in pu.tolist()] if pu is not None else None,
                             "V64": [[round(float(v), 6) for v in row] for row in pv.tolist()] if pv is not None else None,
                             "pooledFull": [[round(float(v), 7) for v in row] for row in
                                            d.reshape(64, m // 64, 64, k // 64).mean((1, 3)).tolist()]}
            del d
        other = {}
        for n, p in sd.items():
            if n.endswith(".weight") and n[:-7] in lin_set:
                continue
            dd = (p.detach().float() - W0[n].to(dev))
            other[n] = {"shape": list(p.shape), "fro": float(dd.norm()), "relChange": float(dd.norm()) / max(float(W0[n].norm()), 1e-12)}
        json.dump({"tag": job["tag"], "linear": spec, "other": other, "pooled": pooled},
                  open(outp.replace("res_", "svd_"), "w"))
        R["spectraFile"] = os.path.basename(outp.replace("res_", "svd_"))
        save()
        if job.get("truncate"):
            # Rebuild the model from the base plus rank-r pieces of each linear delta. Embeddings and norms go
            # back to the base, because all-linear LoRA cannot change them.
            ft = {n: p.detach().to("cpu", copy=True) for n, p in sd.items()}
            res = []
            for r in job["truncate"] + ["full"]:
                with torch.no_grad():
                    for n, p in sd.items():
                        if n.endswith(".weight") and n[:-7] in lin_set:
                            U, S, Vh = tops[n[:-7]]
                            if r == "full":
                                p.copy_(ft[n].to(dev))
                            else:
                                k = min(r, U.shape[1])
                                p.copy_(W0[n].to(dev) + (U[:, :k].to(dev) * S[:k].to(dev)) @ Vh[:k].to(dev))
                        else:
                            p.copy_(W0[n].to(dev))
                e = evals(model, tok, held, wiki, acc_pairs if job.get("truncate_acc") else [], dev, f"trunc r={r}")
                e["rank"] = r
                res.append(e); R["truncation"] = res; save()
    R["warnings"] = logbuf.getvalue()[-2000:]
    R["ok"] = True
except Exception as e:
    R["ok"] = False; R["error"] = f"{type(e).__name__}: {str(e)[:900]}"; R["trace"] = traceback.format_exc()[-2500:]
    try:
        R["peakMemGB"] = gb(torch.cuda.max_memory_allocated)
    except Exception:
        pass
    R["warnings"] = logbuf.getvalue()[-2000:]
save()
print("RESULT", {k: R.get(k) for k in ("tag", "ok", "peakMemGB", "secPerStepMedian", "trainableParams", "error")}, flush=True)

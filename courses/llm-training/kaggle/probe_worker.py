"""One probe job in its own process. Usage: python probe_worker.py <job.json> <result.json>
job: {tag, model, mode: lora|fullft|qlora, steps, batch, time_cap}"""
import gc, json, os, sys, time, traceback, statistics, io, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
job = json.load(open(sys.argv[1])); outp = sys.argv[2]
R = {"tag": job["tag"], "model": job["model"], "mode": job["mode"]}
def save():
    json.dump(R, open(outp, "w"), indent=1)
import torch, transformers, peft
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
logbuf = io.StringIO()
h = logging.StreamHandler(logbuf); h.setLevel(logging.WARNING)
logging.getLogger("transformers").addHandler(h)
def fla_state():
    s = {}
    for m in ("fla", "causal_conv1d"):
        try:
            __import__(m); s[m] = True
        except Exception:
            s[m] = False
    return s
R["fla"] = fla_state()
try:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"][:20000] if len(t.strip()) > 40)[:900000]
    tok = AutoTokenizer.from_pretrained(job["model"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    R["tokensAvailable"] = int(ids.numel())
    T = 1024; B = job["batch"]; N = job["steps"]
    need = B * T * N
    assert ids.numel() >= need, "not enough tokens"
    stream = ids[:need].view(N, B, T)
    t0 = time.time()
    if job["mode"] == "qlora":
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        kw = dict(quantization_config=qcfg, dtype=torch.float16, device_map="cuda:0")
    else:
        kw = dict(dtype=torch.float32, device_map="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(job["model"], **kw)
    R["loadSec"] = round(time.time() - t0, 1)
    R["loadedClass"] = type(model).__name__
    names = [n for n, _ in model.named_modules()]
    R["visionTowerLoaded"] = any(("visual" in n or "vision" in n) for n in names)
    R["modelType"] = getattr(model.config, "model_type", None)
    R["totalParamsBase"] = sum(p.numel() for p in model.parameters())
    R["loadWarnings"] = logbuf.getvalue()[:1500]
    print("loaded", R["loadedClass"], "vision", R["visionTowerLoaded"], flush=True)
    save()

    if job["mode"] == "qlora":
        # prepare_model_for_kbit_training upcasts the 248k-vocab embedding/lm_head to fp32 (3.79 GiB each): OOM on a T4. Do the safe parts by hand.
        for p in model.parameters():
            p.requires_grad = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False

    if job["mode"] in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        ex = r".*(visual|vision).*" if R["visionTowerLoaded"] else None
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", exclude_modules=ex, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        R["loraTargetNames"] = sorted({n.split(".")[-1] for n, m in model.named_modules() if hasattr(m, "lora_A") and not isinstance(m, torch.nn.ModuleDict)})
        tr = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in tr)
        R["totalParams"] = sum(p.numel() for p in model.parameters())
        R["trainableDtypes"] = sorted({str(p.dtype) for p in tr})
        opt = torch.optim.AdamW(tr, lr=2e-4)
    else:
        import bitsandbytes as bnb
        tr = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in tr)
        opt = bnb.optim.AdamW8bit(tr, lr=1e-5)
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    micro = job["batch"]; times = []; losses = []; nonfinite = 0; skipped = 0; capped = False

    def step_fn(batch, micro):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, batch.shape[0], micro):
            x = batch[i:i + micro].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=x).loss
            frac = x.shape[0] / batch.shape[0]
            tot += loss.detach().float().item() * frac
            scaler.scale(loss * frac).backward()
        prev = scaler.get_scale()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(tr, 1.0)
        scaler.step(opt); scaler.update()
        return tot, scaler.get_scale() < prev

    ts = time.time()
    for s in range(N):
        t1 = time.time()
        while True:
            try:
                loss, sk = step_fn(stream[s], micro)
                break
            except torch.OutOfMemoryError:
                R.setdefault("oomRetries", []).append({"step": s, "micro": micro})
                opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                if micro == 1:
                    raise
                micro //= 2
        torch.cuda.synchronize()
        dt = time.time() - t1
        times.append(dt); losses.append(loss); skipped += int(sk)
        if not (loss == loss and abs(loss) != float("inf")):
            nonfinite += 1
        print(f"step {s+1} loss {loss:.4f} {dt:.2f}s scale {scaler.get_scale()}", flush=True)
        if time.time() - ts > job["time_cap"]:
            capped = True
            break
    sel = times[5:] if len(times) > 6 else times[1:]
    med = statistics.median(sel) if sel else None
    R.update(stepsDone=len(times), microBatch=micro, timeCapHit=capped, tokensPerStep=B * T,
             sPerStepMedian=round(med, 3) if med else None,
             tokensPerSec=round(B * T / med, 1) if med else None,
             stepTimes=[round(t, 2) for t in times], loss1=losses[0], lossLast=losses[-1], losses=[round(l, 4) for l in losses],
             nonFiniteLosses=nonfinite, skippedSteps=skipped, gradScalerFinal=scaler.get_scale(),
             peakMemGB=round(torch.cuda.max_memory_allocated() / 1e9, 2), ok=True)
    R["warnings"] = logbuf.getvalue()[:2000]
except Exception as e:
    R["ok"] = False; R["error"] = f"{type(e).__name__}: {str(e)[:900]}"; R["trace"] = traceback.format_exc()[-1800:]
    R["warnings"] = logbuf.getvalue()[:2000]
    try:
        R["peakMemGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
save()

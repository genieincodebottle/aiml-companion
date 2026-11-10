import copy, random
import torch.nn.utils.parametrize as P

BITS, GROUP = 3, 32         # 3-bit weights in groups of 32 columns: round-to-nearest loses clearly at this size
RUN_QAT = DEV.type == "cuda"   # 150 training steps, 593 s on our A100, longer on a T4; needs a GPU

def fake_quant(w, bits=BITS, group=GROUP):
    # Symmetric round-to-nearest per group of columns, returned in float: the values a 3-bit file would hold
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(-1, group)
    scale = (g.abs().amax(1, keepdim=True) / (qmax + 0.5)).clamp_min(torch.finfo(w.dtype).smallest_normal)
    return (torch.clamp(torch.round(g * (1.0 / scale)), -qmax - 1, qmax) * scale).reshape(w.shape)

class FakeQuant(torch.nn.Module):
    def forward(self, w):
        return w + (fake_quant(w) - w).detach()      # straight-through: forward sees the quantized weight, backward sees identity

def linears(m):
    return [mod for name, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and "lm_head" not in name]

def rtn(sd):
    # Post-training quantization: round every linear weight once, no training
    return {k: (fake_quant(v.float()) if v.dim() == 2 and "embed" not in k else v.float()) for k, v in sd.items()}

load_into(rtn(merged_sd))
runs["rtn 3-bit"] = panel(model)
show("RTN 3-bit, no training", runs["rtn 3-bit"])
load_into(merged_sd)

if RUN_QAT:
    # Distil the merged model into a copy that trains with 3-bit weights in the forward pass (LLM-QAT style)
    teacher = model
    student = copy.deepcopy(model).float().train()
    for mod in linears(student):
        P.register_parametrization(mod, "weight", FakeQuant())
    for name, p in student.named_parameters():
        p.requires_grad = "parametrizations" in name          # train the linear weights; embeddings and norms stay
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    hi_rows = [r["text"] for r in load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", streaming=True).skip(64).take(600)]
    uf_rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=2).select(range(600))
    texts = hi_rows + [TEMPLATE.format(e["prompt"]) + e["chosen"][-1]["content"].strip() + "<|endoftext|>" for e in uf_rows]
    random.Random(0).shuffle(texts)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    STEPS, MICRO, ACC = 150, 1, 8          # 8 sequences of up to 512 tokens per step
    t0, hist, i = time.time(), [], 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACC):
            enc = tok(texts[i:i + MICRO], max_length=512, truncation=True, padding=True, return_tensors="pt").to(DEV)
            i = (i + MICRO) % (len(texts) - MICRO)
            with torch.no_grad():
                t_logp = torch.log_softmax(teacher(**enc).logits.float(), -1)
            with torch.autocast("cuda", dtype=torch.float16):
                s_logits = student(**enc).logits
            s_logp = torch.log_softmax(s_logits.float(), -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)                  # forward KL per position
            loss = (kl * enc.attention_mask).sum() / enc.attention_mask.sum() / ACC
            scaler.scale(loss).backward()
            del t_logp, s_logp, s_logits
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        hist.append(round(loss.item() * ACC, 4))
        if step % 25 == 0 or step == STEPS - 1:
            print("step", step, "| KL to the merged model", hist[-1], "|", round(time.time() - t0), "s")
    runs["qat"] = {"klFirst": hist[0], "klLast": hist[-1], "trainSec": round(time.time() - t0, 1),
                   "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "hist": hist}
    for mod in linears(student):
        P.remove_parametrizations(mod, "weight", leave_parametrized=False)   # keep the trained float weights
    qat_sd = rtn({k: v for k, v in student.state_dict().items() if k != "lm_head.weight"})
    del student, opt
    gc.collect()
    torch.cuda.empty_cache()
    load_into(qat_sd)
    runs["qat 3-bit"] = panel(model)
    show("QAT 3-bit", runs["qat 3-bit"])
    print("   QAT minus RTN, Hindi bpb", ci(runs["rtn 3-bit"], runs["qat 3-bit"], "hi"), "| reply NLL",
          ci(runs["rtn 3-bit"], runs["qat 3-bit"], "reply"), "| train", runs["qat"]["trainSec"], "s, peak", runs["qat"]["peakGB"], "GB")
    load_into(merged_sd)

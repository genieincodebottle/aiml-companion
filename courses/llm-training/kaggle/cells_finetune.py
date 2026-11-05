CELLS = [
("md", "# LLM Training Lab: fine-tuning test runs on a free T4\n\nThree short runs, each recorded separately: full SFT of SmolLM2-360M (with our own masked loss checked against TRL's on the same batch), LoRA on Qwen3-1.7B, QLoRA on Qwen3-8B. fp16 throughout (the T4 has no bf16). Writes `llt_finetune_results.json`."),
("code", r'''
import gc, json, os, platform, subprocess, sys, time, traceback
# Kaggle gives 2x T4. Trainer wraps a multi-GPU model in nn.DataParallel, which SFTTrainer refuses,
# so single-GPU labs expose one GPU BEFORE torch is imported (measured failure, run 1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Kaggle's image ships torchao 0.10; PEFT 0.21 refuses to build LoRA layers next to it (measured failure, run 1).
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.22.1", "trl==1.17.1", "peft==0.21.2", "accelerate", "bitsandbytes", "datasets"], check=False)
import torch, transformers, trl, peft, datasets
R = {"date": time.strftime("%Y-%m-%d"), "gpu": torch.cuda.get_device_name(0), "gpuCount": torch.cuda.device_count(),
     "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
             "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__}}
print(R)
def save():
    with open("/kaggle/working/llt_finetune_results.json", "w") as f: json.dump(R, f, indent=1)
def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
'''),
("code", r'''
from datasets import load_dataset, Dataset
# Conversational prompt-completion pairs: the last assistant turn is the completion, so only it carries loss.
stream = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train", streaming=True)
rows = []
for ex in stream:
    msgs = ex["messages"]
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and len(msgs[-1]["content"]) < 3000:
        rows.append({"prompt": msgs[:-1], "completion": [msgs[-1]]})
    if len(rows) >= 4000: break
data = Dataset.from_list(rows)
R["data"] = {"dataset": "HuggingFaceTB/smoltalk smol-magpie-ultra (streamed, first 4000 usable rows)", "rows": len(data)}
print(R["data"])
'''),
("code", r'''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

class Timer(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw): self.t = time.time()
    def on_train_end(self, args, state, control, **kw): self.sec = time.time() - self.t

def run(name, model_id, load_kwargs, cfg_kwargs, peft_cfg=None, kbit=False, check_loss=False, template_from=None):
    free(); out = {"model": model_id}; t = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        out["hadChatTemplate"] = tok.chat_template is not None
        if tok.chat_template is None and template_from:
            # A base model ships without a chat template (measured failure, run 1). Borrow the
            # instruct model's template, and make its end-of-turn token the EOS so the model learns to stop.
            inst = AutoTokenizer.from_pretrained(template_from)
            tok.chat_template = inst.chat_template
            tok.eos_token = inst.eos_token
            out["templateFrom"] = template_from; out["eosToken"] = tok.eos_token
        if tok.pad_token is None or tok.pad_token == tok.eos_token:
            tok.pad_token = tok.unk_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", **load_kwargs)
        if kbit: model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        out["loadSec"] = round(time.time() - t, 1)
        args = SFTConfig(output_dir=f"/kaggle/working/{name}", fp16=True, bf16=False, report_to="none",
                         logging_steps=10, save_strategy="no", completion_only_loss=True, **cfg_kwargs)
        timer = Timer()
        tr = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok, peft_config=peft_cfg, callbacks=[timer])
        out["trainableParams"] = sum(p.numel() for p in tr.model.parameters() if p.requires_grad)
        out["totalParams"] = sum(p.numel() for p in tr.model.parameters())
        if check_loss:
            # Build it, then prove it: our masked NLL must equal the library's loss on the same batch.
            batch = next(iter(tr.get_train_dataloader()))
            batch = {k: v.to("cuda") for k, v in batch.items() if torch.is_tensor(v)}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits = tr.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).logits.float()
                lib = tr.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch["labels"]).loss.float()
            lab = batch["labels"][:, 1:]; lg = logits[:, :-1]
            mask = lab != -100
            own = torch.nn.functional.cross_entropy(lg[mask], lab[mask])
            out["lossCheck"] = {"own": round(own.item(), 5), "library": round(lib.item(), 5),
                                "absDiff": abs(own.item() - lib.item()),
                                "lossTokenFraction": round(mask.float().mean().item(), 3), "batchShape": list(batch["input_ids"].shape)}
            print(out["lossCheck"])
        res = tr.train()
        logs = [h for h in tr.state.log_history if "loss" in h]
        out.update(trainSec=round(timer.sec, 1), steps=tr.state.global_step,
                   secPerStep=round(timer.sec / max(1, tr.state.global_step), 2),
                   lossFirst=logs[0]["loss"] if logs else None, lossLast=logs[-1]["loss"] if logs else None,
                   lossCurve=[{"step": h["step"], "loss": h["loss"]} for h in logs],
                   peakMemGB=round(torch.cuda.max_memory_allocated() / 1e9, 2), ok=True)
        tr.model.eval()
        msgs = [{"role": "user", "content": "Give me two tips for writing a clear email."}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            gen = tr.model.generate(**enc, max_new_tokens=80, do_sample=False)
        out["sample"] = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        out.update(ok=False, error=repr(e)[:600], trace=traceback.format_exc()[-2500:])
        print(out["trace"])
    out["wallClockSec"] = round(time.time() - t, 1)
    R[name] = out; save()
    print(name, {k: v for k, v in out.items() if k not in ("lossCurve", "trace")})
    tr = model = None  # drop references before freeing GPU memory
    free()
'''),
("code", r'''
# 1. Full SFT of SmolLM2-360M base: fp32 master weights, fp16 autocast + GradScaler.
run("sft_smollm2_360m", "HuggingFaceTB/SmolLM2-360M", {"dtype": torch.float32},
    dict(max_length=1024, per_device_train_batch_size=8, gradient_accumulation_steps=2, learning_rate=2e-5,
         max_steps=int(os.environ.get("LLT_SFT_STEPS", "300")), warmup_steps=20, gradient_checkpointing=True),
    check_loss=True, template_from="HuggingFaceTB/SmolLM2-360M-Instruct")
'''),
("code", r'''
# 2. LoRA on Qwen3-1.7B base: fp16 frozen base, adapters on every linear layer.
run("lora_qwen3_1p7b", "Qwen/Qwen3-1.7B-Base", {"dtype": torch.float16},
    dict(max_length=1024, per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4,
         max_steps=int(os.environ.get("LLT_LORA_STEPS", "200")), warmup_steps=10, gradient_checkpointing=True),
    peft_cfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"),
    template_from="Qwen/Qwen3-1.7B")
'''),
("code", r'''
# 3. QLoRA on Qwen3-8B base: NF4 + double quantization, fp16 compute (no bf16 on a T4).
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
run("qlora_qwen3_8b", "Qwen/Qwen3-8B-Base", {"quantization_config": bnb, "dtype": torch.float16},
    dict(max_length=1024, per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
         max_steps=int(os.environ.get("LLT_QLORA_STEPS", "60")), warmup_steps=5, gradient_checkpointing=True),
    peft_cfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"),
    kbit=True, template_from="Qwen/Qwen3-8B")
'''),
("code", r'''
save()
print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("lossCurve", "trace")} if isinstance(v, dict) else v) for k, v in R.items()}, indent=1))
'''),
]

def train_lora(name, tok, rows, seed=0, rank=16, lr=2e-4, out="lora-out", special=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")   # frozen base, fp16
    model.enable_input_require_grads()
    if special:                                          # step 10: give untrained special tokens distinct rows
        E = model.get_input_embeddings().weight
        with torch.no_grad():
            for s in special:                            # the mean of the rows of its spelling, "<" + "tool_call>"
                pieces = tok.encode(s[0], add_special_tokens=False) + tok.encode(s[1:], add_special_tokens=False)
                E[tok.convert_tokens_to_ids(s)] = E[pieces].mean(0)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    args = SFTConfig(output_dir=out, report_to="none", save_strategy="no", seed=seed, data_seed=seed,
                     fp16=True, bf16=False,              # a T4 has no bf16 tensor cores; SFTConfig defaults to bf16
                     learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=10, num_train_epochs=1,
                     per_device_train_batch_size=2, gradient_accumulation_steps=2, max_length=1536,
                     gradient_checkpointing=True, logging_steps=5)
    trainer = SFTTrainer(model=model, args=args, processing_class=tok, train_dataset=Dataset.from_list(rows))
    for p in trainer.model.parameters():                 # GradScaler needs fp32 adapters; re-cast any that are not
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainer.train()
    return trainer

USE_OUR_ADAPTER = False        # True skips training and loads our seed-0 adapter from the Hugging Face Hub
model = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
if USE_OUR_ADAPTER:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    hub = snapshot_download("aimlcompanion/llm-training-lab", allow_patterns=["lltToolUse/qwen3.5-2b-lora-s0/*"])
    base_model = AutoModelForCausalLM.from_pretrained(MAIN, dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base_model, os.path.join(hub, "lltToolUse", "qwen3.5-2b-lora-s0"))
else:
    t0 = time.time()
    trainer = train_lora(MAIN, tok, rows, seed=0)
    train_sec, train_peak = round(time.time() - t0), round(torch.cuda.max_memory_allocated() / 1e9, 2)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    print("steps", trainer.state.global_step, "| loss", logs[0]["loss"], "->", logs[-1]["loss"], "| minutes", round(train_sec / 60, 1),
          "| peak GB", train_peak)
    model = trainer.model
    model.save_pretrained("tooluse-lora-s0")
model.eval()
t0 = time.time()
sft_out = C.generate(model, tok, items, batch_size=32, max_new_tokens=384)
sft_sec = round(time.time() - t0)
sft = C.score(items, sft_out, parse_xml_calls)
print("after LoRA |", json.dumps(C.summary(sft), indent=1))
print("gain and 95% CI:", C.paired_bootstrap([r["correct"] for r in base], [r["correct"] for r in sft]))

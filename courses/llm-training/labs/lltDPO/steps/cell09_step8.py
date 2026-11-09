trl_rows = Dataset.from_list([{"prompt": TEMPLATE.format(x["prompt"]), "chosen": x["chosen"], "rejected": x["rejected"]}
                              for x in dpo_pairs[:320]])
cfg = DPOConfig(output_dir="dpo-trl", report_to="none", fp16=True, bf16=False, beta=0.1, learning_rate=5e-5,
                lr_scheduler_type="cosine", warmup_steps=4, per_device_train_batch_size=2, gradient_accumulation_steps=4,
                max_steps=40, max_length=MAXLEN, gradient_checkpointing=True, precompute_ref_log_probs=True,
                logging_steps=10, save_strategy="no", seed=0)
trainer = DPOTrainer(model=load_sft(), args=cfg, processing_class=tok, train_dataset=trl_rows,
                     peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
kinds = {q.dtype for q in trainer.model.parameters() if q.requires_grad}
for q in trainer.model.parameters():
    if q.requires_grad and q.dtype != torch.float32:
        q.data = q.data.float()                   # fp16 adapter weights break the GradScaler; make them fp32
print("trainable dtypes after the trainer was built", kinds, "| now", {q.dtype for q in trainer.model.parameters() if q.requires_grad})
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer.train()
runs["trl"] = {"steps": 40, "sec": round(time.time() - t0, 1), "sPerStep": round((time.time() - t0) / 40, 2),
               "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "log": [{k: round(v, 4) for k, v in h.items() if isinstance(v, float)} for h in trainer.state.log_history]}
print("DPOTrainer, 40 steps of 8 pairs:", runs["trl"]["sPerStep"], "s per step | peak GB", runs["trl"]["peakGB"])
for h in runs["trl"]["log"][:4]:
    print(h)
del trainer
gc.collect()
torch.cuda.empty_cache()

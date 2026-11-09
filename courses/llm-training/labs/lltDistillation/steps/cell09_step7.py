from datasets import Dataset
from trl import DistillationConfig, DistillationTrainer

# The same objective in TRL's stable trainer: it samples from the student and applies the divergence of step 2 (beta 1 = reverse KL).
ds = Dataset.from_list([{"prompt": prompt(x["q"])} for x in train[:24]])
cfg = DistillationConfig(output_dir="trl-distill", max_steps=3, per_device_train_batch_size=8, learning_rate=LR, beta=1.0,
                         temperature=1.0, max_completion_length=256, fp16=True, bf16=False, gradient_checkpointing=True,
                         logging_steps=1, save_strategy="no", report_to="none",
                         model_init_kwargs={"dtype": "float32"})                   # dtype as a string, not torch.float32
trainer = DistillationTrainer(model=STUDENT, teacher_model=teacher, args=cfg, train_dataset=ds, processing_class=tok,
                              peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                                     task_type="CAUSAL_LM"))
print("trainable dtypes", {str(p.dtype) for p in trainer.model.parameters() if p.requires_grad})
t0 = time.time()
trainer.train()
for h in trainer.state.log_history[:3]:
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items() if k in ("step", "loss", "entropy", "grad_norm")})
print("3 steps in", round(time.time() - t0, 1), "s")
del trainer
gc.collect()
torch.cuda.empty_cache()
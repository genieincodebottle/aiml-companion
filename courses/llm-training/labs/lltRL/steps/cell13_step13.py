RUN_TRL = False     # True: 30 GRPOTrainer steps with the same settings, 12 min on our A100; not yet timed on a T4
from trl import GRPOConfig, GRPOTrainer

def gsm_reward(completions, gold, **kw):
    # TRL passes the dataset's extra columns (here: gold) to every reward function
    return [correct(c, g) for c, g in zip(completions, gold)]

if RUN_TRL:
    ds = Dataset.from_list([dict(prompt=prompt(x["q"]), gold=x["gold"]) for x in train[:30 * 4]])
    args = GRPOConfig(output_dir=os.path.join(WORK, "trl-grpo"), max_steps=30, learning_rate=GRPO["lr"], lr_scheduler_type="constant",
                      per_device_train_batch_size=8, gradient_accumulation_steps=4, num_generations=8,   # 32 replies, 4 prompts per step
                      max_completion_length=MAX_NEW, temperature=1.0, top_k=0, top_p=1.0, beta=0.0, loss_type="dapo",
                      fp16=True, bf16=False,          # bf16 defaults on; a T4 needs fp16. No stop strings: TRL does not pass the tokenizer to generate
                      model_init_kwargs=dict(dtype="float16"), gradient_checkpointing=True,
                      logging_steps=1, save_strategy="no", report_to="none", seed=0)
    trainer = GRPOTrainer(model=GMODEL, reward_funcs=gsm_reward, args=args, train_dataset=ds, processing_class=qtok,
                          peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # GradScaler needs fp32 trainable weights
    print("trainable dtypes", {p.dtype for p in trainer.model.parameters() if p.requires_grad})
    torch.cuda.reset_peak_memory_stats(); t = time.time()
    trainer.train()
    logs = [l for l in trainer.state.log_history if "reward" in l]
    runs["trl-grpo"] = dict(log=logs, sec=round(time.time() - t, 1), peakGB=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    print("30 steps in", runs["trl-grpo"]["sec"], "s | peak GB", runs["trl-grpo"]["peakGB"], "| mean reward",
          round(sum(l["reward"] for l in logs) / len(logs), 3), "| mean completion length", round(sum(l["completions/mean_length"] for l in logs) / len(logs), 1))

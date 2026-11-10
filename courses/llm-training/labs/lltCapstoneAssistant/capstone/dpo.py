"""Stage 4. DPO on the on-policy pairs, TRL DPOTrainer, a fresh LoRA on top of the merged SFT model.

The reference model is the SFT model itself: with a LoRA policy and no ref_model, DPOTrainer gets the
reference log-probs by switching the adapter off, so no second copy sits in memory.
"""
import random

from .common import Timer, cli, done, path, peak_gb, read_jsonl, stage_record, write_json


def main():
    a = cli("Stage 4: DPO", [(("--lr",), dict(type=float, default=2e-5)), (("--beta",), dict(type=float, default=0.1)),
                              (("--epochs",), dict(type=float, default=1.0)), (("--name",), dict(default="dpo"))])
    cfg = a.cfg
    name = a.name
    if done(path(a, name, "merged", "config.json")):
        print("stage 4 already done:", name)
        return
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from .train_common import fp32_trainable, latest_checkpoint, load_model, lora_config, merge_and_save, tokenizer

    sft = path(a, "sft", "merged", "x")[:-2]
    tok = tokenizer(sft)
    pairs = read_jsonl(path(a, "pairs", "pairs.jsonl"))
    random.Random(a.seed).shuffle(pairs)
    n_eval = min(200, len(pairs) // 10) if not a.smoke else max(1, len(pairs) // 3)
    keep = ("prompt", "chosen", "rejected")
    ev = Dataset.from_list([{k: p[k] for k in keep} for p in pairs[:n_eval]])
    tr = Dataset.from_list([{k: p[k] for k in keep} for p in pairs[n_eval:]])
    fp16 = cfg["dtype"] == "float16"
    out_dir = path(a, name, "ckpt", "x")[:-2]
    args = DPOConfig(
        output_dir=out_dir, seed=a.seed, num_train_epochs=a.epochs, max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["dpo_bs"], gradient_accumulation_steps=cfg["dpo_accum"],
        per_device_eval_batch_size=cfg["dpo_bs"], learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        beta=a.beta, loss_type=["sigmoid"], max_length=cfg["dpo_max_len"],
        bf16=not fp16, fp16=fp16, gradient_checkpointing=True,
        logging_steps=2, eval_strategy="steps", eval_steps=25, save_strategy="steps", save_steps=25, save_total_limit=2,
        report_to="none", pad_token=tok.pad_token,
    )
    model = load_model(sft, cfg)
    trainer = DPOTrainer(model=model, args=args, train_dataset=tr, eval_dataset=ev, processing_class=tok,
                         peft_config=lora_config(cfg))
    dtypes = fp32_trainable(trainer.model)
    pre = trainer.evaluate()
    ck = latest_checkpoint(out_dir)
    torch.cuda.reset_peak_memory_stats()
    with Timer() as t:
        trainer.train(resume_from_checkpoint=ck)
    post = trainer.evaluate()
    adapter = path(a, name, "adapter", "x")[:-2]
    trainer.save_model(adapter)
    hist = trainer.state.log_history
    write_json(path(a, name, "log_history.json"), hist)
    peak = peak_gb()
    del trainer, model
    torch.cuda.empty_cache()
    merge_and_save(sft, adapter, path(a, name, "merged", "x")[:-2], cfg)

    def pick(h, keys):
        return {k.replace("eval_", "").replace("rewards/", "r_").replace("logps/", "logp_"): round(h[k], 4)
                for k in keys if k in h}
    kk = ["eval_loss", "eval_rewards/accuracies", "eval_rewards/margins", "eval_rewards/chosen", "eval_rewards/rejected",
          "eval_logps/chosen", "eval_logps/rejected"]
    stage_record(a, name, {
        "wallClockSec": t.sec, "resumedFrom": ck, "peakMemGB": peak, "trainableDtypes": dtypes,
        "config": {"lr": a.lr, "beta": a.beta, "epochs": a.epochs, "batch": cfg["dpo_bs"], "accum": cfg["dpo_accum"],
                   "max_length": cfg["dpo_max_len"], "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"], "loss": "sigmoid",
                   "reference": "SFT model, adapter disabled", "quant": cfg["quant"], "dtype": cfg["dtype"]},
        "n_train": len(tr), "n_heldout": len(ev),
        "heldout": {"before": pick(pre, kk), "after": pick(post, kk)},
        "steps": max([h.get("step", 0) for h in hist] or [0]),
        "train": [{"step": h["step"], **pick(h, ["loss", "rewards/accuracies", "rewards/margins", "logps/chosen", "logps/rejected", "grad_norm"])}
                  for h in hist if "loss" in h],
    })


if __name__ == "__main__":
    main()

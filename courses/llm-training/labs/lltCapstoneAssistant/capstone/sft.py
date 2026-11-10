"""Stage 2. Supervised fine-tuning of the base model with LoRA (T3) or QLoRA (T2), TRL SFTTrainer.

Loss on the reply tokens only (completion_only_loss), rows packed with best-fit decreasing and a per-row position
reset, so attention never crosses from one conversation into the next (transformers builds the block mask).
Resumes from the newest checkpoint if the session stopped.
"""
import json
import math

from .common import BASE, Timer, cli, done, path, peak_gb, read_jsonl, stage_record, write_json


def main():
    a = cli("Stage 2: SFT", [(("--lr",), dict(type=float, default=2e-4)), (("--epochs",), dict(type=float, default=1.0)),
                              (("--name",), dict(default="sft", help="output folder, for an ablation such as sft-noif")),
                              (("--drop-group",), dict(default=None, help="train without one MIX group, such as precise_if"))])
    cfg = a.cfg
    name = a.name
    merged = path(a, name, "merged", "config.json")
    if done(merged):
        print("stage 2 already done")
        return
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from .train_common import fp32_trainable, latest_checkpoint, load_model, lora_config, merge_and_save, tokenizer

    tok = tokenizer(BASE)
    keep = ("prompt", "completion")
    rows = [r for r in read_jsonl(path(a, "data", "sft.jsonl")) if r["group"] != a.drop_group]
    train = Dataset.from_list([{k: r[k] for k in keep} for r in rows])
    heldout = Dataset.from_list([{k: r[k] for k in keep} for r in read_jsonl(path(a, "data", "sft_eval.jsonl"))])
    fp16 = cfg["dtype"] == "float16"
    out_dir = path(a, name, "ckpt", "x")[:-2]
    args = SFTConfig(
        output_dir=out_dir, seed=a.seed, num_train_epochs=a.epochs, max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["sft_bs"], gradient_accumulation_steps=cfg["sft_accum"],
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
        bf16=not fp16, fp16=fp16, gradient_checkpointing=True,
        max_length=cfg["sft_max_len"], packing=True, packing_strategy="bfd", completion_only_loss=True,
        eos_token=tok.eos_token, pad_token=tok.pad_token,
        logging_steps=5, eval_strategy="steps", eval_steps=50, per_device_eval_batch_size=cfg["sft_bs"],
        save_strategy="steps", save_steps=50, save_total_limit=2, report_to="none", dataloader_num_workers=2,
    )
    model = load_model(BASE, cfg)
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=heldout, processing_class=tok,
                         peft_config=lora_config(cfg))
    dtypes = fp32_trainable(trainer.model)
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    pre = trainer.evaluate()                       # held-out loss before training (the base model with the template)
    ck = latest_checkpoint(out_dir)
    torch.cuda.reset_peak_memory_stats()
    with Timer() as t:
        trainer.train(resume_from_checkpoint=ck)
    post = trainer.evaluate()
    adapter = path(a, name, "adapter", "x")[:-2]
    trainer.save_model(adapter)
    hist = trainer.state.log_history
    write_json(path(a, name, "log_history.json"), hist)
    tokens_seen = sum(len(x) for x in trainer.train_dataset["input_ids"]) * a.epochs if "input_ids" in trainer.train_dataset.column_names else None
    del trainer, model
    torch.cuda.empty_cache()
    merge_and_save(BASE, adapter, path(a, name, "merged", "x")[:-2], cfg)
    stage_record(a, name, {
        "dropGroup": a.drop_group,
        "wallClockSec": t.sec, "resumedFrom": ck, "peakMemGB": peak_gb(), "trainableParams": n_trainable, "trainableDtypes": dtypes,
        "config": {"lr": a.lr, "epochs": a.epochs, "batch": cfg["sft_bs"], "accum": cfg["sft_accum"], "max_length": cfg["sft_max_len"],
                   "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"], "quant": cfg["quant"], "dtype": cfg["dtype"],
                   "packing": "bfd", "loss": "completion only"},
        "n_train": len(train), "n_heldout": len(heldout), "tokensPerEpoch": tokens_seen,
        "heldoutLoss": {"before": round(pre["eval_loss"], 4), "after": round(post["eval_loss"], 4)},
        "steps": max([h.get("step", 0) for h in hist] or [0]),
        "trainLoss": [{"step": h["step"], "loss": round(h["loss"], 4), "lr": h.get("learning_rate"), "gradNorm": h.get("grad_norm")}
                      for h in hist if "loss" in h],
        "evalLoss": [{"step": h["step"], "loss": round(h["eval_loss"], 4)} for h in hist if "eval_loss" in h],
    })


if __name__ == "__main__":
    main()

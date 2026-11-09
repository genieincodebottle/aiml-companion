import gc
from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer

uf = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
rows = []
for ex in uf:                                    # the first 4 pairs, in file order, whose two rows fit 320 tokens
    c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
    if c != r and max(len(encode(ex["prompt"], c)[1]), len(encode(ex["prompt"], r)[1])) <= 320:
        rows.append({"prompt": TEMPLATE.format(ex["prompt"]), "chosen": c, "rejected": r, "raw": ex["prompt"]})
    if len(rows) == 4:
        break

policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()   # any policy that differs from the reference
args = DPOConfig(output_dir="dpo-check", report_to="none", use_cpu=True, bf16=False, beta=0.1, per_device_train_batch_size=4)
trainer = DPOTrainer(model=policy, ref_model=cpu_model, args=args, processing_class=tok,
                     train_dataset=Dataset.from_list([{k: x[k] for k in ("prompt", "chosen", "rejected")} for x in rows]))
same = all(list(trainer.train_dataset[i]["prompt_ids"]) == encode(x["raw"], x["chosen"])[0] and
           list(trainer.train_dataset[i]["prompt_ids"]) + list(trainer.train_dataset[i]["chosen_ids"]) == encode(x["raw"], x["chosen"])[1]
           for i, x in enumerate(rows))
print("TRL token ids equal yours, EOS included:", same)

batch = next(iter(trainer.get_train_dataloader()))
n = batch["input_ids"].shape[0] // 2                        # chosen rows first, then rejected rows
full = [batch["input_ids"][i, :int(batch["attention_mask"][i].sum())].tolist() for i in range(2 * n)]
plens = [int(batch["completion_mask"][i].argmax()) for i in range(2 * n)]
with torch.no_grad():
    pi, ref = seq_logps(policy, full, plens), seq_logps(cpu_model, full, plens)
    z = 0.1 * ((pi[:n] - ref[:n]) - (pi[n:] - ref[n:]))
    own = -torch.nn.functional.logsigmoid(z).mean().item()
    lib = trainer.compute_loss(trainer.model, dict(batch)).item()
print("batch", list(batch["input_ids"].shape), "| own", round(own, 6), "| TRL", round(lib, 6), "| difference", abs(own - lib))
print("policy minus reference, chosen", [round(x, 2) for x in (pi[:n] - ref[:n]).tolist()],
      "| rejected", [round(x, 2) for x in (pi[n:] - ref[n:]).tolist()])
del trainer, policy
gc.collect()

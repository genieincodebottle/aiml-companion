from datasets import Dataset
from trl import RewardConfig, RewardTrainer

def as_chat(prompt, reply):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]

rows = [{"chosen": as_chat(p["prompt"], p["chosen"]), "rejected": as_chat(p["prompt"], p["rejected"])} for p in train[:8]]
args = RewardConfig(output_dir="rm-check", report_to="none", use_cpu=True, bf16=False, max_length=MAXLEN,
                    per_device_train_batch_size=4)
trainer = RewardTrainer(model=rm, args=args, processing_class=tok, train_dataset=Dataset.from_list(rows))
print("same token ids as yours:", all(list(trainer.train_dataset[i]["chosen_ids"]) == train[i]["c"] and
                                      list(trainer.train_dataset[i]["rejected_ids"]) == train[i]["r"] for i in range(8)))
batch = next(iter(trainer.get_train_dataloader()))
n = batch["input_ids"].shape[0] // 2                   # TRL stacks the chosen rows first, then the rejected rows
with torch.no_grad():
    r = rm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.squeeze(-1)
    own = -torch.nn.functional.logsigmoid(r[:n] - r[n:]).mean().item()
    lib = trainer.compute_loss(rm, dict(batch)).item()
print("batch", list(batch["input_ids"].shape), "| own", own, "| TRL", lib, "| difference", abs(own - lib))
print("rewards chosen", [round(x, 3) for x in r[:n].tolist()], "| rejected", [round(x, 3) for x in r[n:].tolist()])

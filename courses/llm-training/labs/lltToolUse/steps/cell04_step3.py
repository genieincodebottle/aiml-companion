from datasets import Dataset, Features, Json, List
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

eight = train[:4] + train[N_PER_SOURCE:N_PER_SOURCE + 4]
as_rows = [{"messages": c["messages"], "tools": json.dumps(c["tools"])} for c in eight]
plain = Dataset.from_list(as_rows)                 # Arrow merges every call's argument keys into one struct
print("arguments of the first call after from_list:", plain[0]["messages"][1]["tool_calls"][0]["function"]["arguments"])
typed = Dataset.from_list(as_rows, features=Features({"messages": List(Json()), "tools": Json()}))

dev = "cuda" if torch.cuda.is_available() else "cpu"
small = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.float32).to(dev)
args = SFTConfig(output_dir="check", report_to="none", assistant_only_loss=True, max_length=2048,
                 per_device_train_batch_size=4, bf16=False, use_cpu=(dev == "cpu"))
trainer = SFTTrainer(model=small, args=args, processing_class=tok, train_dataset=typed)
batch = {k: v.to(dev) for k, v in next(iter(trainer.get_train_dataloader())).items()}

ours = torch.full_like(batch["input_ids"], -100)
for r in range(len(ours)):
    n = int(batch["attention_mask"][r].sum())
    ours[r, :n] = torch.tensor(build_labels(batch["input_ids"][r, :n].tolist(), HEADER, IM_END, NEWLINE))
labels_equal = torch.equal(ours, batch["labels"])
print("labels equal:", labels_equal, "| batch", list(batch["input_ids"].shape))

with torch.no_grad():
    total, count = 0.0, 0
    for r in range(len(ours)):                      # one row at a time: a 248k-wide fp32 logit row is about 1 MB per token
        logits = small(input_ids=batch["input_ids"][r:r + 1], attention_mask=batch["attention_mask"][r:r + 1]).logits[0]
        keep = ours[r, 1:] != -100
        total += torch.nn.functional.cross_entropy(logits[:-1][keep].double(), ours[r, 1:][keep], reduction="sum").item()
        count += int(keep.sum())
    lib = trainer.compute_loss(trainer.model, batch).item()
print("own", total / count, "| TRL", lib, "| difference", abs(total / count - lib))
del small, trainer
gc.collect()
torch.cuda.empty_cache()

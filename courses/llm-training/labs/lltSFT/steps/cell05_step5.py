from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# TRL finds assistant tokens through {% generation %} markers in the template. SmolLM2's has none, so add them.
tok.chat_template = inst.chat_template.replace(
    "' + message['content'] + '<|im_end|>' + '",
    "'}}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|im_end|>' }}"
    "{% endgeneration %}{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}{{'")
print("renders the same ids:", list(tok.apply_chat_template(chat)["input_ids"]) == list(inst.apply_chat_template(chat)["input_ids"]))

args = SFTConfig(output_dir="sft-check", report_to="none", use_cpu=True, bf16=False,
                 assistant_only_loss=True, max_length=1024, per_device_train_batch_size=4)
eight = [{"messages": c["messages"]} for c in train[1200:1204] + train[:4]]
trainer = SFTTrainer(model=model, args=args, processing_class=tok, train_dataset=Dataset.from_list(eight))
batch = next(iter(trainer.get_train_dataloader()))

ours = torch.full_like(batch["input_ids"], -100)
for r in range(len(ours)):
    n = int(batch["attention_mask"][r].sum())
    ours[r, :n] = torch.tensor(build_labels(batch["input_ids"][r, :n].tolist()))
print("labels equal:", torch.equal(ours, batch["labels"]), "| batch", list(batch["input_ids"].shape))

with torch.no_grad():
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    target = ours[:, 1:]
    keep = target != -100
    own = torch.nn.functional.cross_entropy(logits[:, :-1][keep].double(), target[keep]).item()
    lib = trainer.compute_loss(trainer.model, dict(batch)).item()
print("own", own, "| TRL", lib, "| difference", abs(own - lib))

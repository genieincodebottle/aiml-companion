from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from lora_lab import MODEL, load_gsm8k, encode, collate

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
ids, mask, labels = collate(encode(tok, load_gsm8k("train", 2)), tok.pad_token_id, "cpu")
batch = dict(input_ids=ids, attention_mask=mask, labels=labels)

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
peft_model = get_peft_model(model, cfg)
torch.manual_seed(1)
with torch.no_grad():
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.normal_(0, 0.02)                           # non-zero B, so the check exercises the adapters
    ref = peft_model(**batch)
n_peft = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
adapters = {name: (m.lora_A["default"].weight.clone(), m.lora_B["default"].weight.clone())
            for name, m in peft_model.base_model.model.named_modules() if hasattr(m, "lora_A")}
print(len(adapters), "layers wrapped by PEFT:", sorted({n.split(".")[-1] for n in adapters}))

model = peft_model.unload()                              # the base model again, weights untouched
add_lora(model, {n.split(".")[-1] for n in adapters}, r=16, alpha=32)
with torch.no_grad():
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.A.copy_(adapters[name][0])
            m.B.copy_(adapters[name][1])
    ours = model(**batch)
n_ours = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable  ours {n_ours:,}  PEFT {n_peft:,}")
print(f"loss       ours {ours.loss.item():.6f}  PEFT {ref.loss.item():.6f}")
print(f"max abs logit difference {(ours.logits - ref.logits).abs().max():.2e}")

for m in model.modules():
    if isinstance(m, LoRALinear):
        m.merge()
with torch.no_grad():
    merged = model(**batch)
print(f"after merge, loss {merged.loss.item():.6f}, max abs logit difference {(merged.logits - ref.logits).abs().max():.2e}")
del model, peft_model, ref, ours, merged                # frees about 3 GB of RAM before the GPU steps

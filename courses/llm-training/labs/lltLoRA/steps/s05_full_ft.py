import random
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_lab import MODEL, load_gsm8k, encode, text_rows, mean_loss, train, accuracy

DEV = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
pairs = load_gsm8k("train")
random.Random(1234).shuffle(pairs)
train_rows = encode(tok, pairs[:2000])                   # 2,000 worked answers, 125 steps of 16
test = load_gsm8k("test")
held, wiki, problems = encode(tok, test[:300]), text_rows(tok), test[:200]

for name in ["model", "opt"]:                          # running this cell again for another FULL_LR:
    globals().pop(name, None)                            # free the last run's model and optimizer first
torch.cuda.empty_cache()

def load():
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map=DEV)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model

def report(name, model, hist=None):
    acc, _, _ = accuracy(model, tok, problems, DEV)
    line = (f"{name:22} held-out loss {mean_loss(model, held, tok.pad_token_id, DEV):.4f} | "
            f"wiki loss {mean_loss(model, wiki, tok.pad_token_id, DEV):.4f} | GSM8K {acc:.1%}")
    if hist:
        line += (f" | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                 f"{sum(h['sec'] for h in hist) / 60:.1f} min | skipped steps {sum(h['skipped'] for h in hist)}")
    print(line)

model = load()
N = sum(p.numel() for p in model.parameters())
print(f"{N:,} parameters | predicted training state: 8-bit AdamW {10 * N / 1e9:.2f} GB, 32-bit AdamW {16 * N / 1e9:.2f} GB")
report("base model", model)

W0 = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}   # for the SVD in step 7
opt = bnb.optim.AdamW8bit(model.parameters(), lr=FULL_LR, weight_decay=0.0)
torch.cuda.reset_peak_memory_stats()
hist = train(model, train_rows, 125, opt, tok.pad_token_id, DEV, budget=FULL_BUDGET)
report("full fine-tune", model, hist)

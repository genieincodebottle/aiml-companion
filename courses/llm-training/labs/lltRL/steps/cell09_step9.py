import math
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GenerationConfig

for name in ["policy", "ref", "value", "rm", "trained"]:     # free the PPO models before loading the GRPO policy
    globals().pop(name, None)
gc.collect()
if DEV.type == "cuda":
    torch.cuda.empty_cache()

GMODEL = "Qwen/Qwen3-0.6B-Base"
N_TEST, MAX_NEW = 200, 256
PASS_AT_K = False     # True: also sample 8 replies per test question for pass@k (12 min more on our shared A100)
qtok = AutoTokenizer.from_pretrained(GMODEL, padding_side="left")
gsm = load_dataset("openai/gsm8k", "main")
test = [dict(q=e["question"], gold=e["answer"].split("####")[-1].strip().replace(",", "")) for e in gsm["test"].select(range(N_TEST))]
train = [dict(q=e["question"], gold=e["answer"].split("####")[-1].strip().replace(",", "")) for e in gsm["train"].shuffle(seed=0)]

def load_policy(adapter=None):
    # frozen fp16 base + fp32 LoRA r16 on every linear layer; adapter: a saved LoRA to continue from
    base = AutoModelForCausalLM.from_pretrained(GMODEL, dtype=torch.float16 if DEV.type == "cuda" else torch.float32).to(DEV)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if adapter:
        model = PeftModel.from_pretrained(base, "aimlcompanion/llm-training-lab", subfolder=adapter, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model

def sample(model, prompts, temperature=1.0, batch=64):
    # completions as token-id lists. A reply ends at a stop string (pad after it is dropped) or at a real end-of-text token (kept).
    greedy = temperature == 0
    cfg = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=not greedy, temperature=None if greedy else temperature,
                           top_k=None if greedy else 0, top_p=None if greedy else 1.0, stop_strings=STOPS,
                           pad_token_id=qtok.pad_token_id, eos_token_id=qtok.eos_token_id)
    model.eval(); model.config.use_cache = True
    out = []
    for i in range(0, len(prompts), batch):
        enc = qtok(prompts[i:i + batch], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad(), autocast():
            g = model.generate(**enc, generation_config=cfg, tokenizer=qtok)
        for row in g[:, enc.input_ids.shape[1]:].tolist():
            n = row.index(qtok.eos_token_id) if qtok.eos_token_id in row else len(row)
            stopped = any(s in qtok.decode(row[:n]) for s in STOPS)
            out.append(row[:n] if stopped or n == len(row) else row[:n + 1])
    model.config.use_cache = False
    return out

def pass_at_k(n, c, k):
    # unbiased estimate of the chance that at least one of k samples is right, from n samples with c right
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def evaluate_gsm(model, k_samples=0):
    t = time.time()
    replies = [qtok.decode(r) for r in sample(model, [prompt(x["q"]) for x in test], temperature=0)]
    right = [int(correct(r, x["gold"])) for r, x in zip(replies, test)]
    res = dict(greedy=round(sum(right) / len(test), 4), right=right,
               meanTokens=round(sum(len(qtok(cut(r)).input_ids) for r in replies) / len(replies), 1), replies=replies[:20])
    if k_samples:
        rows = sample(model, [prompt(x["q"]) for x in test for _ in range(k_samples)])
        c = [int(sum(correct(qtok.decode(rows[i * k_samples + j]), x["gold"]) for j in range(k_samples))) for i, x in enumerate(test)]
        res["nCorrect"] = c
        for k in [1, 2, 4, 8]:
            if k <= k_samples:
                res["pass@" + str(k)] = round(sum(pass_at_k(k_samples, ci, k) for ci in c) / len(test), 4)
    res["sec"] = round(time.time() - t, 1)
    return res

policy = load_policy()
print("trainable", sum(p.numel() for p in policy.parameters() if p.requires_grad), "| dtypes", {p.dtype for p in policy.parameters() if p.requires_grad},
      "| pad id = EOS id:", qtok.pad_token_id == qtok.eos_token_id)
base_eval = evaluate_gsm(policy, 8 if PASS_AT_K else 0)
print("base model on", N_TEST, "GSM8K test questions", {k: v for k, v in base_eval.items() if k != "replies"})
print(repr(base_eval["replies"][0][:300]))

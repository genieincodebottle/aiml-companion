import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # Kaggle has 2 GPUs; this lab uses one
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy, gc, json, random, time
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICY, RM = "HuggingFaceTB/SmolLM2-135M", "lvwerra/distilbert-imdb"
P_LEN, R_LEN = 8, 32                 # every prompt is 8 tokens, every response exactly 32
tok = AutoTokenizer.from_pretrained(POLICY)
rm_tok = AutoTokenizer.from_pretrained(RM)

def gb():
    return round(torch.cuda.memory_allocated() / 1e9, 3) if DEV.type == "cuda" else 0.0

class ValueModel(torch.nn.Module):
    # the same transformer body as the policy, with one number per position instead of a vocabulary
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(POLICY, dtype=torch.float32)
        self.head = torch.nn.Linear(self.body.config.hidden_size, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
    def forward(self, ids):
        h = self.body(input_ids=ids).last_hidden_state
        return self.head(h).squeeze(-1)

def load_models():
    mem = {"start": gb()}
    policy = AutoModelForCausalLM.from_pretrained(POLICY, dtype=torch.float32).to(DEV)        # trained, fp32 master weights
    mem["policy"] = gb()
    ref = AutoModelForCausalLM.from_pretrained(POLICY, dtype=torch.float16 if DEV.type == "cuda" else torch.float32).to(DEV).eval()
    mem["reference"] = gb()
    value = ValueModel().to(DEV)                                                               # trained, fp32
    mem["value"] = gb()
    rm = AutoModelForSequenceClassification.from_pretrained(RM).to(DEV).eval()                 # frozen sentiment classifier
    mem["reward"] = gb()
    for m in (ref, rm):
        m.requires_grad_(False)
    return policy, ref, value, rm, mem

# Prompts: the first 8 tokens of IMDb reviews. Train prompts from the train split, held-out prompts from the test split.
imdb = load_dataset("stanfordnlp/imdb")
def prefixes(split, n, seed):
    out = []
    for text in imdb[split].shuffle(seed=seed)["text"]:
        ids = tok(text[:300].replace("<br />", " ")).input_ids
        if len(ids) >= 2 * P_LEN:
            out.append(ids[:P_LEN])
        if len(out) == n:
            return torch.tensor(out)
train_prompts, eval_prompts = prefixes("train", 12800, 0), prefixes("test", 128, 0)

@torch.no_grad()
def score(seqs, rm):
    # reward = the classifier's log-odds that prompt + response is a positive review
    enc = rm_tok(tok.batch_decode(seqs, skip_special_tokens=True), return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
    logits = rm(**enc).logits.float()
    return logits[:, 1] - logits[:, 0]

@torch.no_grad()
def generate(policy, prompts, seed=None):
    # fp32 sampling at temperature 1, no top-k or top-p, exactly R_LEN new tokens
    if seed is not None:
        torch.manual_seed(seed)
    prompts = prompts.to(DEV)
    return policy.generate(prompts, attention_mask=torch.ones_like(prompts), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                           max_new_tokens=R_LEN, min_new_tokens=R_LEN, pad_token_id=tok.eos_token_id)

def response_logps(model, seqs):
    # log-probs and entropy of the R_LEN response tokens; the logit at position t predicts token t + 1
    logits = model(input_ids=seqs).logits[:, P_LEN - 1:-1].float()
    lp = torch.log_softmax(logits, -1)
    return lp.gather(-1, seqs[:, P_LEN:, None]).squeeze(-1), -(lp.exp() * lp).sum(-1)

policy, ref, value, rm, mem = load_models()
seqs = generate(policy, eval_prompts[:8], seed=0)
s = score(seqs, rm)
for i in range(3):
    print(round(s[i].item(), 2), "|", repr(tok.decode(seqs[i], skip_special_tokens=True)))
print("memory GB after each model", mem, "| response tokens per row", seqs.shape[1] - P_LEN)
print("train prompts", len(train_prompts), "| held-out prompts", len(eval_prompts), "| mean score of 8 samples", round(s.mean().item(), 3))

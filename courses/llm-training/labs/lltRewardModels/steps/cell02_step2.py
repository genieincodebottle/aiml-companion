import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle has 2 GPUs; this lab uses one
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(4)
BASE = "HuggingFaceTB/SmolLM2-360M"
tok = AutoTokenizer.from_pretrained(BASE)
tok.chat_template = AutoTokenizer.from_pretrained(BASE + "-Instruct").chat_template   # the base ships none
tok.pad_token = "<|endoftext|>"                    # padding stays apart from the <|im_end|> that closes a turn

torch.manual_seed(0)
rm = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.float32).eval()
print(rm.score, "| parameters", sum(p.numel() for p in rm.parameters()))

def chat_ids(prompt, response):
    return list(tok.apply_chat_template([{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": response}])["input_ids"])

a = chat_ids("Name a fruit.", "A mango.")
b = chat_ids("Name a fruit.", "A mango is a stone fruit grown in South Asia.")
ids = torch.full((2, len(b)), tok.pad_token_id)
ids[0, :len(a)], ids[1, :len(b)] = torch.tensor(a), torch.tensor(b)
mask = (ids != tok.pad_token_id).long()
print("rows of", len(a), "and", len(b), "tokens, last token of row 0:", repr(tok.decode(a[-1:])))

try:
    rm(input_ids=ids, attention_mask=mask)
except ValueError as e:
    print("ValueError:", e)
rm.config.pad_token_id = tok.pad_token_id
with torch.no_grad():
    rewards = rm(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
    h = rm.model(input_ids=ids, attention_mask=mask).last_hidden_state
    last = mask.sum(-1) - 1                         # the last real token of each right-padded row
    by_hand = rm.score(h[torch.arange(2), last]).squeeze(-1)
print("rewards", rewards.tolist(), "| by hand", by_hand.tolist(), "| equal", torch.allclose(rewards, by_hand))

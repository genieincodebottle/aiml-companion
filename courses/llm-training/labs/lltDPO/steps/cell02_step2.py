import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle has 2 GPUs; this lab uses one
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
BASE = "Qwen/Qwen3-0.6B-Base"
tok = AutoTokenizer.from_pretrained(BASE)
cpu_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()

# The chat-control tokens of this base model share one embedding row that was never trained
E = cpu_model.get_input_embeddings().weight
ctrl = [tok.convert_tokens_to_ids(t) for t in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]]
print("control rows identical:", all(torch.equal(E[ctrl[0]], E[i]) for i in ctrl), "| their norm", round(E[ctrl[0]].norm().item(), 4),
      "| <|endoftext|> norm", round(E[tok.eos_token_id].norm().item(), 4), "| median row norm", round(E.norm(dim=1).median().item(), 4))

NL = chr(10)
TEMPLATE = "### Instruction:" + NL + "{}" + NL + NL + "### Response:" + NL     # plain text, trained tokens only

def encode(prompt, reply):
    # prompt ids, and prompt + reply + <|endoftext|> ids; the prompt must be a prefix of the full row
    p = tok(TEMPLATE.format(prompt)).input_ids
    full = tok(TEMPLATE.format(prompt) + reply).input_ids + [tok.eos_token_id]
    assert full[:len(p)] == p, "the prompt is not a prefix of prompt + reply"
    return p, full

def seq_logps(model, rows, plens, mask_prompt=True):
    # Summed log-prob of each reply. rows: token-id lists; plens: prompt length of each row.
    dev = next(model.parameters()).device
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), tok.pad_token_id, device=dev)
    score = torch.zeros((len(rows), L), dtype=torch.bool, device=dev)
    for i, (r, p) in enumerate(zip(rows, plens)):
        ids[i, :len(r)] = torch.tensor(r, device=dev)
        score[i, (p if mask_prompt else 1):len(r)] = True       # mask from lengths: the pad id is also the EOS id
    att = (torch.arange(L, device=dev)[None] < torch.tensor([len(r) for r in rows], device=dev)[:, None]).long()
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (lp * score[:, 1:]).sum(-1)

p, c = encode("Name a fruit.", "A mango.")
_, r = encode("Name a fruit.", "Fruit is food.")
print("prompt", len(p), "tokens | chosen row", len(c), "| rejected row", len(r), "| pad id = EOS id:", tok.pad_token_id == tok.eos_token_id)
with torch.no_grad():
    masked = seq_logps(cpu_model, [c, r], [len(p)] * 2)
    whole = seq_logps(cpu_model, [c, r], [len(p)] * 2, mask_prompt=False)
print("reply log-probs", [round(x, 3) for x in masked.tolist()], "| with prompt tokens counted", [round(x, 3) for x in whole.tolist()])
print("chosen minus rejected, reply only", round((masked[0] - masked[1]).item(), 4), "| prompt counted", round((whole[0] - whole[1]).item(), 4))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
BASE = "HuggingFaceTB/SmolLM2-360M"
INSTRUCT = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok_b = AutoTokenizer.from_pretrained(BASE)
tok_i = AutoTokenizer.from_pretrained(INSTRUCT)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
inst = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.float32).eval()

@torch.no_grad()
def greedy(model, tok, ids, n):
    # Greedy decoding: always take the most likely next token.
    x = torch.tensor([ids])
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, len(ids):].tolist()

THREE = ["How do I make a cup of green tea?", "Why is the sky blue?", "What is 15 percent of 80?"]
for prompt in THREE:
    ids = tok_b(prompt)["input_ids"]
    print(prompt)
    for name, tok, model in [("base", tok_b, base), ("instruct", tok_i, inst)]:
        print("  ", name.ljust(9), repr(tok.decode(greedy(model, tok, ids, 40))[:90]))

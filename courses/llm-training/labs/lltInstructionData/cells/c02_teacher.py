from transformers import AutoModelForCausalLM

TEACHER = "Qwen/Qwen3-4B-Instruct-2507"   # Apache-2.0, picked by our three-teacher test
N_SYNTH = 300
NL = chr(10)
t_tok = AutoTokenizer.from_pretrained(TEACHER)
t_tok.padding_side = "left"
t_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=t_dtype).to(device).eval()
end_id = t_tok.convert_tokens_to_ids("<|im_end|>")

@torch.no_grad()
def generate(prompts, max_new, sample, batch_size):
    # Returns (text, ended) pairs; ended is False when the reply hit max_new before <|im_end|>.
    outs = []
    for s in range(0, len(prompts), batch_size):
        enc = t_tok(prompts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = teacher.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None,
                             top_p=1.0 if sample else None, top_k=0 if sample else None,
                             eos_token_id=end_id, pad_token_id=t_tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append((t_tok.decode(row, skip_special_tokens=True).strip(), bool((row == end_id).any())))
    return outs

# Magpie: give the teacher only the start of a user turn. It writes a plausible user request itself.
torch.manual_seed(0)
raw = generate(["<|im_start|>user" + NL] * N_SYNTH, max_new=96, sample=True, batch_size=64)
run_on = sum(not ended for _, ended in raw)
# This teacher rarely closes the user turn. It writes the request, a blank line, then starts answering it.
instructions = [text.split(NL + NL)[0].strip() for text, _ in raw]
instructions = [q for q in instructions if len(q) >= 15]

# Then answer each request in the normal chat format, greedy.
chat_prompts = [t_tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                for q in instructions]
answers = generate(chat_prompts, max_new=512, sample=False, batch_size=32)
synth = [{"id": "synth-" + str(k), "source": "synthetic", "sub": TEACHER, "prompt": q, "response": a, "ended": ended}
         for k, (q, (a, ended)) in enumerate(zip(instructions, answers)) if a]
pool += synth
print(N_SYNTH, "requests sampled |", run_on, "never closed the user turn |", len(synth), "pairs kept | pool now", len(pool))
print("answers that ended within 512 tokens:", sum(r["ended"] for r in synth))
for r in synth[:4]:
    print("-", r["prompt"][:110].replace(NL, " "))
del teacher
if torch.cuda.is_available():
    torch.cuda.empty_cache()

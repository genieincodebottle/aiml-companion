gen = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True)
HEADER = list(gen["input_ids"])[-4:]          # the generation prompt is the assistant header
print("assistant header", HEADER, tok.convert_ids_to_tokens(HEADER))
IM_END = tok.convert_tokens_to_ids("<|im_end|>")

def build_labels(ids):
    labels = [-100] * len(ids)                 # -100 means no loss at this position
    t = 0
    while t < len(ids):
        if ids[t:t + 4] == HEADER:              # an assistant reply starts after its header
            t += 4
            while t < len(ids):
                labels[t] = ids[t]              # train on the reply ...
                t += 1
                if ids[t - 1] == IM_END:        # ... up to and including its end-of-turn token
                    break
        else:
            t += 1
    return labels

ex = train[1200]["messages"]                    # the first rewrite example, which has a system prompt
ids = list(tok.apply_chat_template(ex)["input_ids"])
labels = build_labels(ids)
shown = "".join(("[" + tok.decode([i]) + "]") if l != -100 else tok.decode([i]) for i, l in zip(ids, labels))
print(shown[:260], "...", shown[-90:])
for k, split in enumerate(SPLITS):
    n_tok = n_loss = 0
    for c in train[1200 * k:1200 * (k + 1)]:
        ids = list(tok.apply_chat_template(c["messages"])["input_ids"])
        n_tok += len(ids)
        n_loss += sum(l != -100 for l in build_labels(ids))
    print(split, "| tokens", n_tok, "| share with loss", round(n_loss / n_tok, 3))

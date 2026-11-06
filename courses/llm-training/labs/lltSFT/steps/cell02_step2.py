from datasets import load_dataset

def to_messages(ex):
    # SmolTalk2 keeps the system prompt in chat_template_kwargs, not in messages
    msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
    system = (ex.get("chat_template_kwargs") or {}).get("custom_instructions") or ""
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs

SPLITS = ["smoltalk_smollm3_smol_magpie_ultra_no_think", "smoltalk_smollm3_smol_rewrite_no_think",
          "smoltalk_smollm3_smol_summarize_no_think"]

def select(split, n_eval=100, n_train=1200, max_tokens=1024):
    kept = []
    for row, ex in enumerate(load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True)):
        msgs = to_messages(ex)
        n = len(tok.apply_chat_template(msgs)["input_ids"])
        if n <= max_tokens:
            kept.append({"split": split, "row": row, "messages": msgs, "n_tokens": n})
        if len(kept) == n_eval + n_train:
            print(split, "read", row + 1, "rows, kept", len(kept))
            return kept[:n_eval], kept[n_eval:]

held, train = [], []
for split in SPLITS:
    e, t = select(split)
    held += e
    train += t
print(len(train), "training and", len(held), "held-out conversations")
print("system prompts lost without to_messages:", sum(m["messages"][0]["role"] == "system" for m in train))

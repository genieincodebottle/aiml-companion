"""Shared code for the lltSFT lab: data selection, the training chat template, the label mask and the evals.

Used by measure.py (CPU, T1) and worker_sft.py (Kaggle T4, T2), so every run selects the same conversations
and scores them the same way. The lab page shows the same functions in its build steps.

Data: HuggingFaceTB/smoltalk2, config SFT, three no_think splits rebuilt from SmolTalk's Smol-Magpie-Ultra,
Smol-rewrite and Smol-summarize (Apache-2.0 per the SmolTalk card). Streamed in file order, so the selection is
deterministic. A conversation is kept when it renders to at most MAX_TOKENS tokens under the SmolLM2 template.
The first N_EVAL kept rows of each split are the held-out eval set, the next N_TRAIN the training set.
"""
import hashlib
import itertools
import json
import math

import torch

BASE_360 = "HuggingFaceTB/SmolLM2-360M"
BASE_135 = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_360 = "HuggingFaceTB/SmolLM2-360M-Instruct"
INSTRUCT_135 = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATASET = "HuggingFaceTB/smoltalk2"
SPLITS = [
    "smoltalk_smollm3_smol_magpie_ultra_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
]
MAX_TOKENS = 1024
N_EVAL = 100
N_TRAIN = 1200
N_GEN = 20          # first N_GEN eval rows of each split also give a generation prompt


def to_messages(ex):
    """SmolTalk2 stores the system prompt in chat_template_kwargs.custom_instructions, not in messages."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
    system = (ex.get("chat_template_kwargs") or {}).get("custom_instructions") or ""
    if system and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs


def training_template(instruct_template):
    """The instruct template with {% generation %} markers around each assistant reply and its <|im_end|>.

    Renders the same text as the instruct template. TRL's assistant_only_loss needs the markers.
    """
    t = instruct_template.replace(
        "' + message['content'] + '<|im_end|>' + '",
        "'}}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|im_end|>' }}"
        "{% endgeneration %}{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}{{'")
    assert t != instruct_template, "the instruct template changed; update training_template"
    return t


def setup_tokenizer(tok, instruct_tok):
    """Base SmolLM2 ships no chat template: borrow the instruct one, make <|im_end|> the EOS, keep pad apart from EOS."""
    tok.chat_template = training_template(instruct_tok.chat_template)
    tok.eos_token = "<|im_end|>"
    tok.pad_token = "<|endoftext|>"
    return tok


def header_ids(tok):
    """The assistant header, <|im_start|> assistant newline: the last 4 ids of the generation prompt (ASCII-only source)."""
    gen = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True)
    return list(gen["input_ids"])[-4:]


def build_labels(ids, header, im_end):
    """-100 everywhere except each assistant reply and its <|im_end|>."""
    labels = [-100] * len(ids)
    t = 0
    while t < len(ids):
        if ids[t:t + len(header)] == header:
            t += len(header)
            while t < len(ids):
                labels[t] = ids[t]
                t += 1
                if ids[t - 1] == im_end:
                    break
        else:
            t += 1
    return labels


def select_conversations(tok, n_eval=N_EVAL, n_train=N_TRAIN, max_tokens=MAX_TOKENS, splits=SPLITS, log=print):
    """Returns (train, eval, stats). Each item: {"split", "row", "messages", "n_tokens"}."""
    from datasets import load_dataset
    train, held, stats = [], [], {}
    for split in splits:
        ds = load_dataset(DATASET, "SFT", split=split, streaming=True)
        kept, seen = [], 0
        for row, ex in enumerate(ds):
            seen += 1
            msgs = to_messages(ex)
            n = len(tok.apply_chat_template(msgs)["input_ids"])
            if n <= max_tokens:
                kept.append({"split": split, "row": row, "messages": msgs, "n_tokens": n})
                if len(kept) == n_eval + n_train:
                    break
        held += kept[:n_eval]
        train += kept[n_eval:]
        stats[split] = {"rowsRead": seen, "kept": len(kept), "keptShare": round(len(kept) / seen, 4)}
        log(split, stats[split])
    return train, held, stats


def data_hash(items):
    h = hashlib.sha256()
    for it in items:
        h.update(json.dumps(it["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@torch.no_grad()
def eval_loss(model, tok, convs, device, batch_size=8, dtype=None):
    """Token-weighted masked NLL over the assistant tokens of each held-out conversation, one conversation per row."""
    header, im_end = header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    was_training = model.training
    model.eval()
    nll, count = 0.0, 0
    for i in range(0, len(convs), batch_size):
        chunk = convs[i:i + batch_size]
        ids = [list(tok.apply_chat_template(c["messages"])["input_ids"]) for c in chunk]
        T = max(len(x) for x in ids)
        x = torch.full((len(ids), T), tok.pad_token_id, dtype=torch.long)
        y = torch.full((len(ids), T), -100, dtype=torch.long)
        am = torch.zeros((len(ids), T), dtype=torch.long)
        for r, row in enumerate(ids):
            x[r, :len(row)] = torch.tensor(row)
            y[r, :len(row)] = torch.tensor(build_labels(row, header, im_end))
            am[r, :len(row)] = 1
        x, y, am = x.to(device), y.to(device), am.to(device)
        with torch.autocast(device.type if hasattr(device, "type") else str(device), dtype=dtype, enabled=dtype is not None):
            logits = model(input_ids=x, attention_mask=am, use_cache=False).logits
        tgt = y[:, 1:]
        m = tgt != -100
        nll += torch.nn.functional.cross_entropy(logits[:, :-1][m].float(), tgt[m], reduction="sum").item()
        count += int(m.sum())
    if was_training:
        model.train()
    return {"loss": nll / count, "tokens": count}


@torch.no_grad()
def eval_generate(model, tok, convs, device, max_new_tokens=512, batch_size=20, dtype=None, template=True):
    """Greedy answers to the first user turn (with its system prompt). Stop rate = share that end with <|im_end|>."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    was_training = model.training
    model.eval()
    prompts = []
    for c in convs:
        msgs = c["messages"]
        first = [m for m in msgs[:2] if m["role"] == "system"] + [next(m for m in msgs if m["role"] == "user")]
        if template:
            prompts.append(list(tok.apply_chat_template(first, add_generation_prompt=True)["input_ids"]))
        else:
            prompts.append(list(tok("\n\n".join(m["content"] for m in first) + "\n\n")["input_ids"]))
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        T = max(len(p) for p in chunk)
        x = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(chunk), T), dtype=torch.long)
        for r, p in enumerate(chunk):          # left padding for generation
            x[r, T - len(p):] = torch.tensor(p)
            am[r, T - len(p):] = 1
        x, am = x.to(device), am.to(device)
        with torch.autocast(device.type if hasattr(device, "type") else str(device), dtype=dtype, enabled=dtype is not None):
            g = model.generate(input_ids=x, attention_mask=am, max_new_tokens=max_new_tokens, do_sample=False,
                               eos_token_id=im_end, pad_token_id=tok.pad_token_id)
        for r in range(len(chunk)):
            new = g[r, T:].tolist()
            stopped = im_end in new
            n = new.index(im_end) + 1 if stopped else len([t for t in new if t != tok.pad_token_id])
            out.append({"stopped": stopped, "newTokens": n, "text": tok.decode(new[:n], skip_special_tokens=False)})
    if was_training:
        model.train()
    stopped = [o for o in out if o["stopped"]]
    return {
        "n": len(out), "stopRate": round(len(stopped) / len(out), 4),
        "meanNewTokens": round(sum(o["newTokens"] for o in out) / len(out), 1),
        "meanNewTokensStopped": round(sum(o["newTokens"] for o in stopped) / len(stopped), 1) if stopped else None,
        "samples": out,
    }


def generation_set(held, n_per_split=N_GEN, splits=SPLITS):
    return [c for s in splits for c in [h for h in held if h["split"] == s][:n_per_split]]


def cross_doc_collator(collate, restart_positions, explicit_mask=True):
    # Same packed rows and labels as TRL's collator, but every token may attend to every earlier token in the row.
    # explicit_mask=True passes a full causal mask, so SDPA gets an explicit mask, as it does with the document mask.
    def wrapped(examples):
        b = collate(examples)
        n = b["input_ids"].shape[1]
        if not restart_positions:
            b["position_ids"] = torch.arange(n)[None]
        if explicit_mask:
            b["attention_mask"] = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"):
            b.pop(k, None)
        return b
    return wrapped

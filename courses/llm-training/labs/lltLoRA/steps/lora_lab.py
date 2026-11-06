# lora_lab.py - data, training loop and evaluation for the lltLoRA lab.
# The notebook writes this file; our recorded runs (kaggle/lora_worker.py) import the same file.
import math
import re
import time

import torch
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-0.8B-Base"
PROMPT = """Question: {q}
Answer:"""


def load_gsm8k(split, n=None):
    rows = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None:
        rows = rows.select(range(n))
    # Drop the calculator notes such as <<48/2=24>> from each worked answer.
    return [(PROMPT.format(q=r["question"]), " " + re.sub("<<[^>]*>>", "", r["answer"])) for r in rows]


def encode(tok, pairs):
    rows = []
    for prompt, answer in pairs:
        p = tok(prompt)["input_ids"]
        a = tok(answer)["input_ids"] + [tok.eos_token_id]
        rows.append((p + a, [-100] * len(p) + a))      # loss only on the answer and its end token
    return rows


def text_rows(tok, n_rows=32, T=512):
    # General text to measure forgetting: Wikipedia articles the task never shows.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("".join(ds["text"]))["input_ids"][: n_rows * T]
    return [(ids[i:i + T], ids[i:i + T]) for i in range(0, n_rows * T, T)]


def micro_batches(rows, budget):
    # Group rows so that rows x longest row stays within budget tokens. Memory follows that product,
    # because the logits tensor is rows x length x 248,320 vocabulary entries.
    rows = sorted(rows, key=lambda r: len(r[0]))
    groups, cur = [], []
    for r in rows:
        if cur and (len(cur) + 1) * len(r[0]) > budget:
            groups.append(cur)
            cur = []
        cur.append(r)
    return groups + [cur]


def collate(rows, pad_id, device):
    T = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), T), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def amp(device):
    return torch.autocast(device, dtype=torch.float16, enabled=device == "cuda")


@torch.no_grad()
def mean_loss(model, rows, pad_id, device, budget=2048):
    # Mean next-token loss over every label that is not -100, in nats per token.
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for group in micro_batches(rows, budget):
        ids, mask, labels = collate(group, pad_id, device)
        k = int((labels[:, 1:] != -100).sum())
        with amp(device):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=k).loss
        total += loss.item() * k
        n += k
    model.train(was_training)
    return total / n


def train(model, rows, steps, opt, pad_id, device="cuda", batch=16, budget=1024, warmup=10, seed=0,
          on_step=None):
    params = [p for p in model.parameters() if p.requires_grad]
    # Linear warmup over the first warmup steps, then linear decay to 0 at the last step.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * (1 - s / steps))
    # Qwen3.5's fp16 gradients overflow at the default scale of 65,536, so start lower.
    scaler = torch.amp.GradScaler(device, init_scale=2048, enabled=device == "cuda")
    g = torch.Generator().manual_seed(seed)
    order, history = [], []
    model.train()
    for step in range(steps):
        t0 = time.time()
        if len(order) < batch:
            order += torch.randperm(len(rows), generator=g).tolist()
        picked = [rows[i] for i in order[:batch]]
        order = order[batch:]
        n_tok = sum(sum(1 for t in y if t != -100) for _, y in picked)
        loss_sum = 0.0
        for group in micro_batches(picked, budget):
            ids, mask, labels = collate(group, pad_id, device)
            with amp(device):
                # num_items_in_batch makes the loss a sum over this micro-batch divided by the
                # answer tokens of the whole batch, so every token weighs the same.
                loss = model(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=n_tok).loss
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scale = scaler.get_scale()
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if device == "cuda":
            torch.cuda.synchronize()
        rec = {"step": step + 1, "trainLoss": round(loss_sum, 5), "gradNorm": round(gnorm, 4),
               "lr": sched.get_last_lr()[0], "skipped": scaler.get_scale() < scale,
               "sec": round(time.time() - t0, 3), "tokens": n_tok}
        history.append(rec)
        if on_step:
            on_step(rec)
    return history


NUMBER = "-?[0-9][0-9,]*(?:[.][0-9]+)?"


def final_number(text, strict=True):
    # strict: the number after ####, the answer format of the training data.
    # lenient: that number, or else the last number before the model starts a new question.
    text = text.split("Question:")[0]
    m = re.search("####[ ]*(" + NUMBER + ")", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    if strict:
        return None
    found = re.findall(NUMBER, text)
    return found[-1].replace(",", "").rstrip(".") if found else None


@torch.no_grad()
def accuracy(model, tok, pairs, device="cuda", batch=50, max_new=256):
    # Greedy answers; a problem counts as solved when the number after #### matches the reference.
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    strict, lenient, outputs = 0, 0, []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok([p for p, _ in chunk], return_tensors="pt", padding=True).to(device)
        with amp(device):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id)
        for (_, ref), seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            text = tok.decode(seq, skip_special_tokens=True)
            gold = final_number(ref)
            strict += final_number(text) == gold
            lenient += final_number(text, strict=False) == gold
            outputs.append(text)
    tok.padding_side = "right"
    model.train(was_training)
    return strict / len(pairs), lenient / len(pairs), outputs

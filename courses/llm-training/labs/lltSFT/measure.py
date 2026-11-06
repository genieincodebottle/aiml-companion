"""Exact T1 measurements for the lltSFT page (CPU, .venv-latest). Each command writes one Run Library file.

  python measure.py data       lltSFT-t1-data       selection, token counts, loss-token share, padding vs packing
  python measure.py template   lltSFT-t1-template   chat template, default system prompt, pad and EOS facts
  python measure.py masktrl    lltSFT-t1-mask-trl   own label mask and masked NLL vs TRL SFTTrainer on SmolLM2-135M
  python measure.py packing    lltSFT-t1-packing    packed logits vs alone, with and without the document mask
  python measure.py truncation lltSFT-t1-truncation how many conversations lose their last <|im_end|> at each max_length

Timings are not page figures here (the CPU is shared); measure_timing.py gives those.
"""
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sft_common as C  # noqa: E402
from runlog import peak_rss_gb, write_run  # noqa: E402

torch.set_num_threads(int(os.environ.get("LLT_THREADS", "4")))


def tokenizers(base=C.BASE_135, inst=C.INSTRUCT_135):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    it = AutoTokenizer.from_pretrained(inst)
    return C.setup_tokenizer(tok, it), it


def cache_path():
    return os.path.join(HERE, "data_cache.json")


def load_selection(tok):
    import json
    p = cache_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d["train"], d["eval"], d["stats"]
    train, held, stats = C.select_conversations(tok)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"train": train, "eval": held, "stats": stats}, f, ensure_ascii=False)
    return train, held, stats


def cmd_data():
    t0 = time.time()
    tok, _ = tokenizers()
    train, held, stats = load_selection(tok)
    header, im_end = C.header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    per_split = {}
    for split in C.SPLITS:
        rows = [c for c in train if c["split"] == split]
        ids = [list(tok.apply_chat_template(c["messages"])["input_ids"]) for c in rows]
        lab = [C.build_labels(x, header, im_end) for x in ids]
        n_tok = sum(len(x) for x in ids)
        n_loss = sum(sum(l != -100 for l in y) for y in lab)
        per_split[split] = {"train": len(rows), "tokens": n_tok, "meanTokens": round(n_tok / len(rows), 1),
                            "lossTokens": n_loss, "lossShare": round(n_loss / n_tok, 4),
                            "assistantTurnsMean": round(sum(sum(m["role"] == "assistant" for m in c["messages"]) for c in rows) / len(rows), 2),
                            "withSystem": sum(c["messages"][0]["role"] == "system" for c in rows)}
    ids = [list(tok.apply_chat_template(c["messages"])["input_ids"]) for c in train]
    lab = [C.build_labels(x, header, im_end) for x in ids]
    lens = [len(x) for x in ids]
    n_tok, n_loss = sum(lens), sum(sum(l != -100 for l in y) for y in lab)
    # Padding: shuffle with seed 0, batches of 4 conversations padded to the longest
    order = list(range(len(lens)))
    random.Random(0).shuffle(order)
    padded = 0
    for i in range(0, len(order), 4):
        b = [lens[j] for j in order[i:i + 4]]
        padded += max(b) * len(b)
    # Packing: TRL's best-fit decreasing into rows of 1,024
    from datasets import Dataset
    from trl import pack_dataset
    packed = pack_dataset(Dataset.from_dict({"input_ids": ids}), seq_length=C.MAX_TOKENS, strategy="bfd")
    rows = len(packed)
    seqs_per_row = [len(s) for s in packed["seq_lengths"]]
    evals = {
        "selection": stats, "perSplit": per_split,
        "train": {"conversations": len(train), "tokens": n_tok, "meanTokens": round(n_tok / len(train), 1),
                  "lossTokens": n_loss, "lossShare": round(n_loss / n_tok, 4)},
        "eval": {"conversations": len(held), "tokens": sum(c["n_tokens"] for c in held)},
        "padded4": {"positions": padded, "padShare": round(1 - n_tok / padded, 4), "lossShareOfPositions": round(n_loss / padded, 4)},
        "packedBfd1024": {"rows": rows, "positions": rows * C.MAX_TOKENS, "fill": round(n_tok / (rows * C.MAX_TOKENS), 4),
                          "meanConversationsPerRow": round(sum(seqs_per_row) / rows, 2), "maxConversationsPerRow": max(seqs_per_row),
                          "lossShareOfPositions": round(n_loss / (rows * C.MAX_TOKENS), 4)},
        "trainHash": C.data_hash(train), "evalHash": C.data_hash(held),
    }
    print(evals)
    write_run("lltSFT-t1-data", "T1", "baseline", {"dataset": C.DATASET, "config": "SFT", "splits": C.SPLITS,
              "maxTokens": C.MAX_TOKENS, "nEvalPerSplit": C.N_EVAL, "nTrainPerSplit": C.N_TRAIN, "tokenizer": C.BASE_135},
              None, [{"step": 0, "peakRssGB": peak_rss_gb()}], evals=evals, wall_clock_sec=round(time.time() - t0, 1))


def cmd_template():
    from transformers import AutoTokenizer
    raw = AutoTokenizer.from_pretrained(C.BASE_135)
    inst = AutoTokenizer.from_pretrained(C.INSTRUCT_135)
    inst360 = AutoTokenizer.from_pretrained(C.INSTRUCT_360)
    base360 = AutoTokenizer.from_pretrained(C.BASE_360)
    tok, _ = tokenizers()
    msgs = [{"role": "user", "content": "Name a fruit."}, {"role": "assistant", "content": "A mango."}]
    with_default = list(inst.apply_chat_template(msgs)["input_ids"])
    with_system = list(inst.apply_chat_template([{"role": "system", "content": "Be brief."}] + msgs)["input_ids"])
    # tokens the default system block adds: render a system message with the default text vs no system
    default_text = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    explicit = list(inst.apply_chat_template([{"role": "system", "content": default_text}] + msgs)["input_ids"])
    ev = {
        "base135HasTemplate": raw.chat_template is not None, "base360HasTemplate": base360.chat_template is not None,
        "baseEos": [raw.eos_token, raw.eos_token_id], "basePad": [raw.pad_token, raw.pad_token_id],
        "instructEos": [inst.eos_token, inst.eos_token_id], "instructPad": [inst.pad_token, inst.pad_token_id],
        "instructPadIsEos": inst.pad_token_id == inst.eos_token_id,
        "sameTemplate135and360": inst.chat_template == inst360.chat_template,
        "sameVocab": raw.get_vocab() == inst.get_vocab(), "vocab": len(raw),
        "trainingTemplateRendersSameIds": list(tok.apply_chat_template(msgs)["input_ids"]) == with_default,
        "exampleTokens": len(with_default), "exampleTokensWithShortSystem": len(with_system),
        "defaultEqualsExplicit": explicit == with_default,
        "headerIds": C.header_ids(tok), "headerTokens": tok.convert_ids_to_tokens(C.header_ids(tok)),
        "imEnd": tok.convert_tokens_to_ids("<|im_end|>"), "endoftext": tok.convert_tokens_to_ids("<|endoftext|>"),
        "newlineToken": tok.convert_ids_to_tokens(tok("x\ny")["input_ids"]),
    }
    # The default system block, measured directly: ids before the first <|im_start|>user
    us = tok.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t in enumerate(with_default) if t == us]
    ev["defaultSystemBlockTokens"] = starts[1]
    ev["defaultSystemBlockShareOfExample"] = round(starts[1] / len(with_default), 4)
    ev["renderedExample"] = inst.apply_chat_template(msgs, tokenize=False)
    print(ev)
    write_run("lltSFT-t1-template", "T1", "baseline", {"base": C.BASE_135, "instruct": C.INSTRUCT_135}, None,
              [{"step": 0}], evals=ev)


def cmd_masktrl():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    t0 = time.time()
    tok, inst = tokenizers()
    train, held, _ = load_selection(tok)
    convs = [c["messages"] for c in train if c["split"] == C.SPLITS[1]][:4] + [c["messages"] for c in train if c["split"] == C.SPLITS[0]][:4]
    header, im_end = C.header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(C.BASE_135, dtype=torch.float32)
    ev = {}
    # 1. Stock instruct template: assistant_only_loss refuses it
    stock = AutoTokenizer.from_pretrained(C.BASE_135)
    stock.chat_template, stock.eos_token, stock.pad_token = inst.chat_template, "<|im_end|>", "<|endoftext|>"
    try:
        SFTTrainer(model=model, args=SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none", bf16=False,
                   use_cpu=True, assistant_only_loss=True), train_dataset=Dataset.from_list([{"messages": convs[0]}]),
                   processing_class=stock)
        ev["stockTemplateError"] = None
    except Exception as e:
        ev["stockTemplateError"] = type(e).__name__ + ": " + str(e)
    # 2. Training template: labels and loss match ours
    args = SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none", bf16=False, use_cpu=True,
                     assistant_only_loss=True, max_length=1024, per_device_train_batch_size=4, shuffle_dataset=False)
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list([{"messages": c} for c in convs]),
                    processing_class=tok)
    batch = next(iter(tr.get_train_dataloader()))
    ids = batch["input_ids"]
    ours = torch.full_like(ids, -100)
    for r in range(ids.shape[0]):
        n = int(batch["attention_mask"][r].sum())
        ours[r, :n] = torch.tensor(C.build_labels(ids[r, :n].tolist(), header, im_end))
    ev["labelsEqual"] = bool(torch.equal(ours, batch["labels"]))
    ev["batchShape"] = list(ids.shape)
    ev["lossTokenShare"] = round(float((batch["labels"][:, 1:] != -100).sum() / batch["attention_mask"][:, 1:].sum()), 4)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=batch["attention_mask"], use_cache=False).logits
        lp = torch.log_softmax(logits[:, :-1].double(), dim=-1)
        tgt = ours[:, 1:]
        m = tgt != -100
        own = -lp[m].gather(-1, tgt[m][:, None]).mean().item()
        trl_loss = tr.compute_loss(tr.model, {k: v for k, v in batch.items()}).item()
        hf = model(input_ids=ids, attention_mask=batch["attention_mask"], labels=ours, use_cache=False).loss.item()
    ev.update(ownLoss=own, trlLoss=trl_loss, hfLoss=hf, absDiffOwnTrl=abs(own - trl_loss), absDiffOwnHf=abs(own - hf),
              lossTokens=int(m.sum()))
    try:
        SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none")
        ev["defaultConfigOnCpu"] = "ok"
    except Exception as e:
        ev["defaultConfigOnCpu"] = type(e).__name__ + ": " + str(e)
    d = SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none", use_cpu=True)
    ev["sftConfigDefaults"] = {"bf16": d.bf16, "fp16": d.fp16, "gradientCheckpointing": d.gradient_checkpointing,
                               "learningRate": d.learning_rate, "maxLength": d.max_length, "packing": d.packing,
                               "packingStrategy": d.packing_strategy, "completionOnlyLoss": d.completion_only_loss,
                               "assistantOnlyLoss": d.assistant_only_loss, "lossType": d.loss_type, "loggingSteps": d.logging_steps}
    print(ev)
    write_run("lltSFT-t1-mask-trl", "T1", "baseline", {"model": C.BASE_135, "dtype": "float32", "conversations": 8,
              "trl": "SFTConfig(assistant_only_loss=True, max_length=1024, per_device_train_batch_size=4)"}, None,
              [{"step": 0, "peakRssGB": peak_rss_gb()}], evals=ev, wall_clock_sec=round(time.time() - t0, 1))


def cmd_packing():
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    tok, _ = tokenizers()
    train, held, _ = load_selection(tok)
    header, im_end = C.header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    convs = [c["messages"] for c in train if c["split"] == C.SPLITS[1]][:3]
    docs = [list(tok.apply_chat_template(c)["input_ids"]) for c in convs]
    lens = [len(d) for d in docs]
    model = AutoModelForCausalLM.from_pretrained(C.BASE_135, dtype=torch.float32).eval()
    ids = torch.tensor([sum(docs, [])])
    T = ids.shape[1]
    pos_reset = torch.cat([torch.arange(n) for n in lens])[None]
    pos_cont = torch.arange(T)[None]
    doc_of = torch.cat([torch.full((n,), k) for k, n in enumerate(lens)])
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    block = causal & (doc_of[:, None] == doc_of[None, :])

    def doc_loss(logits, k, start):
        lab = torch.tensor(C.build_labels(docs[k], header, im_end))
        lg = logits[start:start + lens[k]]
        tgt = lab[1:]
        m = tgt != -100
        return torch.nn.functional.cross_entropy(lg[:-1][m].double(), tgt[m]).item()

    with torch.no_grad():
        alone = [model(input_ids=torch.tensor([d]), use_cache=False).logits[0] for d in docs]
        variants = {
            "positionReset": model(input_ids=ids, position_ids=pos_reset, use_cache=False).logits[0],
            "blockMask4d": model(input_ids=ids, attention_mask=block[None, None], position_ids=pos_reset, use_cache=False).logits[0],
            "naive": model(input_ids=ids, position_ids=pos_cont, use_cache=False).logits[0],
            "positionResetNoMask": model(input_ids=ids, attention_mask=causal[None, None], position_ids=pos_reset, use_cache=False).logits[0],
            "positionResetWithCache": model(input_ids=ids, position_ids=pos_reset).logits[0],
        }
    res = {}
    for name, lg in variants.items():
        start, rows = 0, []
        for k, n in enumerate(lens):
            rows.append({"doc": k + 1, "maxAbsLogitDiff": round((lg[start:start + n] - alone[k]).abs().max().item(), 6),
                         "lossPacked": round(doc_loss(lg, k, start), 4), "lossAlone": round(doc_loss(alone[k], k, 0), 4)})
            start += n
        res[name] = rows
    ev = {"lens": lens, "total": T, "attnImplementation": model.config._attn_implementation,
          "useCacheDefault": model.config.use_cache, "variants": res,
          "packedPositionIds": pos_reset[0].tolist(), "docOf": doc_of.tolist()}
    for k, v in res.items():
        print(k, v)
    write_run("lltSFT-t1-packing", "T1", "baseline", {"model": C.BASE_135, "dtype": "float32", "docs": 3,
              "source": C.SPLITS[1] + " training rows 0-2"}, None, [{"step": 0, "peakRssGB": peak_rss_gb()}],
              evals=ev, samples=[{"doc": k + 1, "ids": d, "labels": C.build_labels(d, header, im_end)} for k, d in enumerate(docs)],
              wall_clock_sec=round(time.time() - t0, 1))


def cmd_truncation():
    tok, _ = tokenizers()
    from datasets import load_dataset
    import itertools
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    header = C.header_ids(tok)
    lens = {}
    for split in C.SPLITS:
        ds = load_dataset(C.DATASET, "SFT", split=split, streaming=True)
        L = []
        for ex in itertools.islice(ds, 2000):
            L.append(len(tok.apply_chat_template(C.to_messages(ex))["input_ids"]))
        lens[split] = L
    out = {}
    for ml in [512, 1024, 2048]:
        out[str(ml)] = {s: round(sum(l > ml for l in L) / len(L), 4) for s, L in lens.items()}
    print(out)
    write_run("lltSFT-t1-truncation", "T1", "baseline", {"rowsPerSplit": 2000, "splits": C.SPLITS, "tokenizer": C.BASE_135},
              None, [{"step": 0}], evals={"shareLongerThanMaxLength": out,
                                          "note": "a conversation longer than max_length loses its last <|im_end|> under keep_start truncation"})


def naive_collator(collate):
    """Wrap TRL's padding-free collator: same rows and labels, positions that never reset, so no document mask."""
    def wrapped(examples):
        b = collate(examples)
        b["position_ids"] = torch.arange(b["input_ids"].shape[1])[None]
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"):
            b.pop(k, None)
        return b
    return wrapped


def cmd_trlpack():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    t0 = time.time()
    tok, _ = tokenizers()
    train, held, _ = load_selection(tok)
    convs = [c["messages"] for c in train if c["split"] == C.SPLITS[1]][:6]
    header, im_end = C.header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(C.BASE_135, dtype=torch.float32)
    args = SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none", bf16=False, use_cpu=True,
                     assistant_only_loss=True, packing=True, max_length=1024, per_device_train_batch_size=1)
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list([{"messages": c} for c in convs]),
                    processing_class=tok)
    batch = next(iter(tr.get_train_dataloader()))
    pos = batch["position_ids"][0]
    starts = (pos == 0).nonzero().flatten().tolist() + [pos.numel()]
    ids = batch["input_ids"][0].tolist()
    model.eval()
    with torch.no_grad():
        trl_masked = tr.compute_loss(tr.model, dict(batch)).item()
        nb = naive_collator(tr.data_collator)
        # rebuild the same examples through the naive collator
        rows = [tr.train_dataset[i] for i in range(len(tr.train_dataset))]
        b2 = nb(rows[:1])
        same_ids = torch.equal(b2["input_ids"], batch["input_ids"]) and torch.equal(b2["labels"], batch["labels"])
        trl_naive = tr.compute_loss(tr.model, dict(b2)).item()
        nll, cnt = 0.0, 0
        for a, b in zip(starts[:-1], starts[1:]):
            doc = ids[a:b]
            lab = torch.tensor(batch["labels"][0, a:b].tolist())
            lg = model(input_ids=torch.tensor([doc]), use_cache=False).logits[0]
            tgt = lab[1:]
            m = tgt != -100
            nll += torch.nn.functional.cross_entropy(lg[:-1][m].double(), tgt[m], reduction="sum").item()
            cnt += int(m.sum())
    ev = {"rowTokens": len(ids), "docsInRow": len(starts) - 1, "docLengths": [b - a for a, b in zip(starts[:-1], starts[1:])],
          "keys": sorted(k for k in batch.keys()), "trlPackedLoss": trl_masked, "ownSeparateLoss": nll / cnt,
          "absDiff": abs(trl_masked - nll / cnt), "naiveSameIdsAndLabels": bool(same_ids), "trlNaiveLoss": trl_naive,
          "lossTokens": cnt}
    print(ev)
    write_run("lltSFT-t1-trl-packing", "T1", "baseline", {"model": C.BASE_135, "dtype": "float32",
              "trl": "SFTConfig(assistant_only_loss=True, packing=True, packing_strategy='bfd', max_length=1024)",
              "conversations": 6, "source": C.SPLITS[1] + " training rows 0-5"}, None,
              [{"step": 0, "peakRssGB": peak_rss_gb()}], evals=ev, wall_clock_sec=round(time.time() - t0, 1))


def cmd_crossdoc():
    """CPU checks for the step 10 arms: TRL packed batch through C.cross_doc_collator, and conversation 2 packed
    after a similar (rewrite) or an unrelated (Magpie) conversation."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    t0 = time.time()
    tok, _ = tokenizers()
    train, held, _ = load_selection(tok)
    header, im_end = C.header_ids(tok), tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(C.BASE_135, dtype=torch.float32)
    rew = [c["messages"] for c in train if c["split"] == C.SPLITS[1]]
    mag = [c["messages"] for c in train if c["split"] == C.SPLITS[0]]
    args = SFTConfig(output_dir=os.path.join(HERE, "_trl_tmp"), report_to="none", bf16=False, use_cpu=True,
                     assistant_only_loss=True, packing=True, max_length=1024, per_device_train_batch_size=1)
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list([{"messages": c} for c in rew[:6]]),
                    processing_class=tok)
    rows = [tr.train_dataset[i] for i in range(len(tr.train_dataset))]
    row = max(rows, key=lambda r: len(r["seq_lengths"]))
    model.eval()
    arms = {}
    with torch.no_grad():
        base = tr.data_collator([row])
        arms["documentMask"] = tr.compute_loss(tr.model, dict(base)).item()
        for name, restart, explicit in [("continuousNoMaskObject", False, False), ("continuousExplicitCausal", False, True),
                                        ("restartExplicitCausal", True, True)]:
            b = C.cross_doc_collator(tr.data_collator, restart, explicit)([row])
            arms[name] = tr.compute_loss(tr.model, dict(b)).item()
        x, pos = base["input_ids"], base["position_ids"]
        n = x.shape[1]
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
        lg = model(input_ids=x, position_ids=pos, attention_mask=causal, use_cache=False).logits[0, :-1]
        tgt = base["labels"][0, 1:]
        m = tgt != -100
        direct = torch.nn.functional.cross_entropy(lg[m].double(), tgt[m]).item()

        def part_loss(first, second, positions):
            d1 = list(tok.apply_chat_template(first)["input_ids"])
            d2 = list(tok.apply_chat_template(second)["input_ids"])
            ids = torch.tensor([d1 + d2])
            if positions == "alone":
                logits = model(input_ids=torch.tensor([d2]), use_cache=False).logits[0]
                off = 0
            else:
                p = torch.arange(len(d1) + len(d2))[None] if positions == "continuous" else torch.tensor([list(range(len(d1))) + list(range(len(d2)))])
                full = torch.tril(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool))[None, None]
                logits = model(input_ids=ids, position_ids=p, attention_mask=full, use_cache=False).logits[0]
                off = len(d1)
            y = torch.tensor(C.build_labels(d2, header, im_end))
            keep = y[1:] != -100
            return round(torch.nn.functional.cross_entropy(logits[off:off + len(d2) - 1][keep].double(), y[1:][keep]).item(), 4)

        ctx = {}
        for k, target, similar in [("conversation2", rew[1], rew[0]), ("conversation3", rew[2], rew[1])]:
            ctx[k] = {
                "alone": part_loss(similar, target, "alone"),
                "afterSimilarContinuous": part_loss(similar, target, "continuous"),
                "afterSimilarRestart": part_loss(similar, target, "restart"),
                "afterUnrelatedContinuous": part_loss(mag[0], target, "continuous"),
                "afterUnrelatedRestart": part_loss(mag[0], target, "restart"),
            }
    ev = {"rowDocLengths": list(row["seq_lengths"]), "trlBatchLosses": arms, "directRestartCausal": direct,
          "absDiffRestartArmVsDirect": abs(arms["restartExplicitCausal"] - direct),
          "absDiffContinuousArms": abs(arms["continuousNoMaskObject"] - arms["continuousExplicitCausal"]),
          "context": ctx, "unrelated": "first Magpie training conversation", "similar": "the rewrite conversation before it"}
    print(ev)
    write_run("lltSFT-t1-crossdoc", "T1", "ablation:packing", {"model": C.BASE_135, "dtype": "float32",
              "collator": "sft_common.cross_doc_collator", "rows": C.SPLITS[1] + " training rows 0-5"}, None,
              [{"step": 0, "peakRssGB": peak_rss_gb()}], evals=ev, wall_clock_sec=round(time.time() - t0, 1))


if __name__ == "__main__":
    {"data": cmd_data, "template": cmd_template, "masktrl": cmd_masktrl, "packing": cmd_packing,
     "truncation": cmd_truncation, "trlpack": cmd_trlpack, "crossdoc": cmd_crossdoc}[sys.argv[1]]()

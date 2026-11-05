"""Measurements behind the lltBaseVsInstruct lab page (LLM Training Lab, module 10).

Every figure the page quotes comes from a Run Library file this script writes to
../../runs/lltBaseVsInstruct/. CPU only, float32, greedy decoding, 4 threads.

Usage:
    python measure.py side        # identical prompts to base and instruct (step 1)
    python measure.py template    # the chat template and its special tokens (step 2)
    python measure.py kl smollm2  # per-token KL on 40 prompts, SmolLM2-360M pair (steps 3-4)
    python measure.py kl qwen3    # same protocol, Qwen3-0.6B-Base vs Qwen3-0.6B (open question)
    python measure.py broken      # template mismatch and missing generation prompt (Diagnose it)
    python measure.py wrong       # instruct fed the raw prompt, base fed the chat template (Diagnose it)
    python measure.py checkpoints # file sizes, dtypes and parameter counts from the safetensors headers
"""
import math
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS  # noqa: E402
from runlog import SUFFIX, peak_rss_gb, read_run, write_run  # noqa: E402

THREADS = int(os.environ.get("LLT_THREADS", "4"))
torch.set_num_threads(THREADS)

PAIRS = {
    "smollm2": ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", {}),
    "qwen3": ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B", {"enable_thinking": False}),
}
MAX_NEW = 128
PLAIN = "Question: {p}\nAnswer:\n"
ALPACA = "### Instruction:\n{p}\n\n### Response:\n"

# Token categories for "where does the KL sit". Rules apply in this order to the decoded token.
DISCOURSE = set("""sure certainly absolutely here however overall additionally furthermore moreover
finally remember hello hi thank thanks great please sorry unfortunately instead also ultimately
conclusion summary note hope enjoy happy glad let okay ok well so first firstly second secondly
lastly next then alternatively therefore thus example simply""".split())
FUNCTION = set("""the a an of to in on at for with by from and or but is are was were be been being
it its this that these those as if not can will would should could do does did you your i me my we
our they their them he she his her there which who what when how why than more most some any each
all has have had into about up out no yes just only very""".split())
FORMAT_CHARS = set("#*-_|`>:")


def category(tok_str, is_end):
    if is_end:
        return "end"
    s = tok_str.strip()
    if s == "" or all(c in FORMAT_CHARS for c in s):
        return "format"
    if any(c.isdigit() for c in s):
        return "number"
    if not any(c.isalnum() for c in s):
        return "punct"
    w = s.lower()
    if w in DISCOURSE:
        return "discourse"
    if w in FUNCTION:
        return "function"
    return "content"


CATS = ["discourse", "function", "content", "number", "punct", "format", "end"]
BUCKETS = [(1, 1), (2, 5), (6, 20), (21, 60), (61, MAX_NEW)]


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    return tok, model


def chat_ids(tok, prompt, kw, gen_prompt=True):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=gen_prompt, **kw)
    return list(enc["input_ids"])


@torch.no_grad()
def greedy(model, tok, ids, n):
    x = torch.tensor([ids])
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n,
                         do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return out[0, len(ids):].tolist()


@torch.no_grad()
def logprobs(model, ctx, resp):
    """Log-probabilities (float64) at the positions that predict each response token."""
    x = torch.tensor([ctx + resp])
    logits = model(input_ids=x).logits[0, len(ctx) - 1: len(ctx) - 1 + len(resp)]
    return torch.log_softmax(logits.double(), dim=-1)


def kl_and_rank(lp_inst, lp_base, resp):
    """Per token: KL(instruct || base) in nats, and the base rank of the token instruct emitted."""
    p = lp_inst.exp()
    kl = (p * (lp_inst - lp_base)).sum(-1)
    idx = torch.tensor(resp)
    tok_lp = lp_base[torch.arange(len(resp)), idx]
    rank = (lp_base > tok_lp[:, None]).sum(-1) + 1
    return kl.tolist(), rank.tolist(), tok_lp.tolist()


def summarise(rows):
    """rows: dicts with t, kl, rank, cat. Returns the aggregate table the page quotes."""
    kl = np.array([r["kl"] for r in rows])
    rank = np.array([r["rank"] for r in rows])
    total = kl.sum()
    out = {
        "tokens": int(len(rows)),
        "meanKL": round(float(kl.mean()), 4),
        "unshifted": round(float((rank == 1).mean()), 4),
        "marginal": round(float(((rank > 1) & (rank <= 3)).mean()), 4),
        "shifted": round(float((rank > 3).mean()), 4),
        "klShareShifted": round(float(kl[rank > 3].sum() / total), 4),
    }
    top = np.sort(kl)[::-1]
    k = max(1, int(round(0.05 * len(kl))))
    out["klShareTop5pct"] = round(float(top[:k].sum() / total), 4)
    out["byBucket"] = {}
    for lo, hi in BUCKETS:
        m = np.array([lo <= r["t"] <= hi for r in rows])
        if m.any():
            out["byBucket"][f"{lo}-{hi}"] = {
                "tokens": int(m.sum()), "tokenShare": round(float(m.mean()), 4),
                "meanKL": round(float(kl[m].mean()), 4), "klShare": round(float(kl[m].sum() / total), 4),
                "shifted": round(float((rank[m] > 3).mean()), 4),
            }
    out["byCategory"] = {}
    for c in CATS:
        m = np.array([r["cat"] == c for r in rows])
        if m.any():
            out["byCategory"][c] = {
                "tokens": int(m.sum()), "tokenShare": round(float(m.mean()), 4),
                "meanKL": round(float(kl[m].mean()), 4), "klShare": round(float(kl[m].sum() / total), 4),
                "shifted": round(float((rank[m] > 3).mean()), 4),
            }
    return out


def top_shifted(rows, n=15):
    counts = {}
    for r in rows:
        if r["rank"] > 3 and r["cat"] != "end":
            key = r["tok"].strip() or repr(r["tok"])
            c = counts.setdefault(key, [0, 0.0])
            c[0] += 1
            c[1] += r["kl"]
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:n]
    return [{"token": k, "count": v[0], "meanKL": round(v[1] / v[0], 3)} for k, v in ranked]


def per_position(rows, upto=40):
    out = []
    for t in range(1, upto + 1):
        ks = [r["kl"] for r in rows if r["t"] == t]
        if ks:
            out.append({"t": t, "n": len(ks), "meanKL": round(float(np.mean(ks)), 4)})
    return out


def cmd_side():
    t0 = time.perf_counter()
    base_id, inst_id, kw = PAIRS["smollm2"]
    tb, base = load(base_id)
    ti, inst = load(inst_id)
    prompts = [PROMPTS[0], PROMPTS[8], PROMPTS[32]]
    samples = []
    for p in prompts:
        raw = tb(p)["input_ids"]
        assert raw == ti(p)["input_ids"], "tokenizers disagree on the raw prompt"
        chat = chat_ids(ti, p, kw)
        for fmt, ids in [("raw", raw), ("chat", chat)]:
            for label, tok, model in [("base", tb, base), ("instruct", ti, inst)]:
                out = greedy(model, tok, ids, 40)
                samples.append({"prompt": p, "format": fmt, "model": label,
                                "completion": tok.decode(out, skip_special_tokens=False)})
                print(fmt, label, "|", repr(samples[-1]["completion"][:160]))
    write_run("lltBaseVsInstruct-t1-side-by-side", "T1", "baseline",
              {"base": base_id, "instruct": inst_id, "dtype": "float32", "decoding": "greedy",
               "max_new_tokens": 40, "threads": 4, "prompts": [PROMPTS.index(p) for p in prompts]},
              None, [{"step": 0, "peakRssGB": peak_rss_gb()}], samples=samples,
              wall_clock_sec=round(time.perf_counter() - t0, 1))


def cmd_template():
    t0 = time.perf_counter()
    info = {}
    for pair, (base_id, inst_id, kw) in PAIRS.items():
        tb = AutoTokenizer.from_pretrained(base_id)
        ti = AutoTokenizer.from_pretrained(inst_id)
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        enc = ti.apply_chat_template(msgs, add_generation_prompt=True, **kw)
        no_gen = ti.apply_chat_template(msgs, add_generation_prompt=False, **kw)
        ids = list(enc["input_ids"])
        info[pair] = {
            "returnType": type(enc).__name__,
            "keys": list(enc.keys()),
            "text": ti.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw),
            "textNoGenPrompt": ti.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw),
            "nTokens": len(ids),
            "nTokensNoGenPrompt": len(list(no_gen["input_ids"])),
            "tokens": ti.convert_ids_to_tokens(ids),
            "ids": ids,
            "special": {k: [v, ti.convert_tokens_to_ids(v)] for k, v in ti.special_tokens_map.items() if isinstance(v, str)},
            "baseSpecial": {k: [v, tb.convert_tokens_to_ids(v)] for k, v in tb.special_tokens_map.items() if isinstance(v, str)},
            "baseHasChatTemplate": tb.chat_template is not None,
            "sameVocab": tb.get_vocab() == ti.get_vocab(),
            "vocab": len(ti),
        }
        print(pair, info[pair]["returnType"], info[pair]["keys"], info[pair]["nTokens"], info[pair]["nTokensNoGenPrompt"])
        print(repr(info[pair]["text"]))
        print(info[pair]["special"], info[pair]["baseSpecial"], "base template:", info[pair]["baseHasChatTemplate"])
    write_run("lltBaseVsInstruct-t1-template", "T1", "baseline", {"pairs": {k: v[:2] for k, v in PAIRS.items()}},
              None, [], evals=info, wall_clock_sec=round(time.perf_counter() - t0, 1))


def cmd_kl(pair):
    t0 = time.perf_counter()
    base_id, inst_id, kw = PAIRS[pair]
    tb, base = load(base_id)
    ti, inst = load(inst_id)
    end_ids = {ti.eos_token_id}
    rows_chat, rows_plain, samples = [], [], []
    gen_tokens, t_gen = 0, 0.0
    for i, p in enumerate(PROMPTS):
        ctx = chat_ids(ti, p, kw)
        g0 = time.perf_counter()
        resp = greedy(inst, ti, ctx, MAX_NEW)
        t_gen += time.perf_counter() - g0
        gen_tokens += len(resp)
        lp_i = logprobs(inst, ctx, resp)
        lp_b = logprobs(base, ctx, resp)
        plain = tb(PLAIN.format(p=p))["input_ids"]
        lp_bp = logprobs(base, plain, resp)
        kl_c, rank_c, _ = kl_and_rank(lp_i, lp_b, resp)
        kl_p, rank_p, _ = kl_and_rank(lp_i, lp_bp, resp)
        nll_inst = float(-lp_i[torch.arange(len(resp)), torch.tensor(resp)].mean())
        toks = [ti.decode([r]) for r in resp]
        per_tok = []
        for t, (r, s) in enumerate(zip(resp, toks), start=1):
            cat = category(s, r in end_ids)
            rows_chat.append({"t": t, "kl": kl_c[t - 1], "rank": rank_c[t - 1], "cat": cat, "tok": s})
            rows_plain.append({"t": t, "kl": kl_p[t - 1], "rank": rank_p[t - 1], "cat": cat, "tok": s})
            per_tok.append([s, round(kl_c[t - 1], 4), rank_c[t - 1], round(kl_p[t - 1], 4), rank_p[t - 1], cat])
        samples.append({"i": i, "prompt": p, "response": ti.decode(resp), "nTokens": len(resp),
                        "ended": resp[-1] in end_ids, "responseIds": resp, "instructNLL": round(nll_inst, 4),
                        "perToken": per_tok})
        print(i, len(resp), round(float(np.mean(kl_c)), 3), round(float(np.mean(kl_p)), 3), repr(ti.decode(resp)[:70]), flush=True)
    no_end = [r for r in rows_chat if r["cat"] != "end"]
    no_end_p = [r for r in rows_plain if r["cat"] != "end"]
    evals = {
        "promptSet": "prompts.py, 40 prompts, 5 groups of 8",
        "chatContext": summarise(rows_chat),
        "chatContextNoEnd": summarise(no_end),
        "plainContextNoEnd": summarise(no_end_p),
        "topShiftedChat": top_shifted(rows_chat),
        "topShiftedPlain": top_shifted(rows_plain),
        "perPositionChat": per_position(rows_chat),
        "perPositionPlainNoEnd": per_position(no_end_p),
        "responses": {"ended": sum(s["ended"] for s in samples), "meanTokens": round(float(np.mean([s["nTokens"] for s in samples])), 1)},
    }
    metrics = [{"step": 0, "peakRssGB": peak_rss_gb(), "generatedTokens": gen_tokens,
                "genSecShared": round(t_gen, 1)}]
    write_run(f"lltBaseVsInstruct-t1-{pair}-kl", "T1", "baseline",
              {"base": base_id, "instruct": inst_id, "chatTemplateKwargs": kw, "dtype": "float32",
               "decoding": "greedy", "max_new_tokens": MAX_NEW, "threads": THREADS, "nPrompts": len(PROMPTS),
               "klDirection": "KL(instruct || base), nats, full vocabulary, float64",
               "baseContexts": {"chat": "same token ids as instruct", "plain": PLAIN},
               "categories": "category() in measure.py; discourse and function word lists there"},
              None, metrics, evals=evals, samples=samples, wall_clock_sec=round(time.perf_counter() - t0, 1))


def cmd_broken():
    t0 = time.perf_counter()
    _, inst_id, kw = PAIRS["smollm2"]
    ti, inst = load(inst_id)
    ref = read_run("lltBaseVsInstruct-t1-smollm2-kl")["samples"]
    conds = {
        "correct": lambda p: chat_ids(ti, p, kw, gen_prompt=True),
        "noGenPrompt": lambda p: chat_ids(ti, p, kw, gen_prompt=False),
        "alpaca": lambda p: ti(ALPACA.format(p=p))["input_ids"],
    }
    res = {}
    for name, make in conds.items():
        nlls, first_nll, outs = [], [], []
        for s in ref:
            ctx = make(s["prompt"])
            resp = s["responseIds"]
            lp = logprobs(inst, ctx, resp)
            tok_lp = lp[torch.arange(len(resp)), torch.tensor(resp)]
            nlls.append(float(-tok_lp.mean()))
            first_nll.append(float(-tok_lp[0]))
            if name != "correct":
                out = greedy(inst, ti, ctx, 48)
                outs.append({"prompt": s["prompt"], "completion": ti.decode(out)})
        r = {"meanNLL": round(float(np.mean(nlls)), 4), "meanFirstTokenNLL": round(float(np.mean(first_nll)), 4)}
        if outs:
            c = [o["completion"] for o in outs]
            r["startsWithImStart"] = sum(x.startswith("<|im_start|>") for x in c)
            r["startsWithAssistantHeader"] = sum(x.startswith("<|im_start|>assistant") for x in c)
            r["containsInstructionHeader"] = sum("### Instruction" in x for x in c)
            r["endsInFirst3Tokens"] = sum(len(ti(x)["input_ids"]) <= 3 for x in c)
            r["samples"] = outs
        res[name] = r
        print(name, {k: v for k, v in r.items() if k != "samples"}, flush=True)
    for name in ("noGenPrompt", "alpaca"):
        write_run(f"lltBaseVsInstruct-t1-broken-{'no-genprompt' if name == 'noGenPrompt' else 'alpaca-template'}",
                  "T1", f"broken:{name}",
                  {"model": inst_id, "dtype": "float32", "decoding": "greedy", "max_new_tokens": 48,
                   "threads": THREADS, "nPrompts": len(ref),
                   "context": "apply_chat_template(add_generation_prompt=False)" if name == "noGenPrompt" else ALPACA,
                   "reference": "greedy responses of run lltBaseVsInstruct-t1-smollm2-kl" + SUFFIX},
                  None, [{"step": 0, "meanNLL": res[name]["meanNLL"], "meanFirstTokenNLL": res[name]["meanFirstTokenNLL"]}],
                  evals={"broken": {k: v for k, v in res[name].items() if k != "samples"},
                         "correct": res["correct"]},
                  samples=res[name]["samples"], wall_clock_sec=round(time.perf_counter() - t0, 1))


@torch.no_grad()
def first_token(model, ids, k=5):
    lp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0, -1].double(), -1)
    top = lp.topk(k)
    return [(int(i), round(float(v.exp()), 4)) for v, i in zip(top.values, top.indices)], lp


def cmd_wrong_format():
    """Each model fed the other's format: instruct gets the raw prompt, base gets the chat template."""
    t0 = time.perf_counter()
    base_id, inst_id, kw = PAIRS["smollm2"]
    tb, base = load(base_id)
    ti, inst = load(inst_id)
    eos = ti.eos_token_id
    inst_raw, base_chat = [], []
    for p in PROMPTS:
        raw = tb(p)["input_ids"]
        top, lp = first_token(inst, raw)
        inst_raw.append({"prompt": p, "pEndFirst": round(float(lp[eos].exp()), 4),
                         "top": [[ti.convert_ids_to_tokens(i), q] for i, q in top],
                         "endsAtOnce": top[0][0] == eos})
        chat = chat_ids(ti, p, kw)
        out = greedy(base, tb, chat, 24)
        q_ids = tb(p)["input_ids"]
        base_chat.append({"prompt": p, "completion": tb.decode(out),
                          "firstTokens": tb.convert_ids_to_tokens(out[:8]),
                          "repeatsQuestion": out[:len(q_ids)] == q_ids,
                          "invalidUtf8Tokens": sum(tb.decode([o]).count(chr(0xFFFD)) > 0 for o in out)})
        print(p[:30], inst_raw[-1]["pEndFirst"], inst_raw[-1]["endsAtOnce"], base_chat[-1]["repeatsQuestion"],
              base_chat[-1]["invalidUtf8Tokens"], flush=True)
    evals = {
        "instructRaw": {"n": len(PROMPTS), "endsAtOnce": sum(r["endsAtOnce"] for r in inst_raw),
                        "meanPEndFirst": round(float(np.mean([r["pEndFirst"] for r in inst_raw])), 4),
                        "medianPEndFirst": round(float(np.median([r["pEndFirst"] for r in inst_raw])), 4)},
        "baseChat": {"n": len(PROMPTS), "repeatsQuestion": sum(r["repeatsQuestion"] for r in base_chat),
                     "withInvalidUtf8Token": sum(r["invalidUtf8Tokens"] > 0 for r in base_chat)},
    }
    print(evals)
    write_run("lltBaseVsInstruct-t1-wrong-format", "T1", "broken:wrong-format",
              {"base": base_id, "instruct": inst_id, "dtype": "float32", "decoding": "greedy",
               "baseMaxNewTokens": 24, "threads": THREADS, "nPrompts": len(PROMPTS),
               "instructContext": "tokenizer(prompt), no chat template",
               "baseContext": "the instruct tokenizer's apply_chat_template(add_generation_prompt=True)"},
              None, [{"step": 0, "peakRssGB": peak_rss_gb()}], evals=evals,
              samples={"instructRaw": inst_raw, "baseChat": base_chat},
              wall_clock_sec=round(time.perf_counter() - t0, 1))


def cmd_checkpoints():
    """Checkpoint file sizes, dtypes and parameter counts, read from the safetensors headers (no weights loaded)."""
    import json as _json
    import struct
    from huggingface_hub import hf_hub_download
    info = {}
    for pair, (base_id, inst_id, _) in PAIRS.items():
        for name in (base_id, inst_id):
            path = hf_hub_download(name, "model.safetensors")
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = _json.loads(fh.read(n))
            header.pop("__metadata__", None)
            info[name] = {
                "fileBytes": os.path.getsize(path),
                "headerBytes": 8 + n,
                "parameters": sum(math.prod(v["shape"]) for v in header.values()),
                "dtypes": sorted({v["dtype"] for v in header.values()}),
                "tensors": len(header),
            }
            print(name, info[name], flush=True)
    write_run("lltBaseVsInstruct-t1-checkpoints", "T1", "baseline", {"source": "safetensors headers from the Hub cache"},
              None, [], evals=info)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "checkpoints":
        cmd_checkpoints()
    elif cmd == "wrong":
        cmd_wrong_format()
    elif cmd == "side":
        cmd_side()
    elif cmd == "template":
        cmd_template()
    elif cmd == "kl":
        cmd_kl(sys.argv[2])
    elif cmd == "broken":
        cmd_broken()

"""Exact measurements for the lltGeneration page (no timings; those are in measure_timing.py).

Needs out/gen-model.pt and out/tokenizer.json from train_small.py. CPU, float32, 4 threads.
Writes one Run Library file per measurement group into runs/lltGeneration/:

  lltGeneration-t1-sampling       step 3, temperature, top-k, top-p on the trained model; greedy vs top-p repetition
  lltGeneration-t1-cache-exact    step 5, cached vs uncached logits over 200 greedy tokens (max abs error)
  lltGeneration-t1-kv-bytes       step 6, KV formula vs cache tensor bytes, ours and SmolLM2-135M's real HF cache
  lltGeneration-t1-gqa            step 8, cache bytes and allclose for 6, 3, 2, 1 KV heads
  lltGeneration-t1-broken-rope-offset     Diagnose it, RoPE positions restart at 0 under the cache
  lltGeneration-t1-broken-prefill-mask    Diagnose it, no causal mask in the prefill chunk
  lltGeneration-t1-gotchas        help-panel errors: cache dtype mismatch, is_causal=True on a decode step

    python measure.py --threads 4
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F

from model import GPT, load
from runlog import write_run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PROMPT = "Once upon a time, there was a little girl named Lily. She"


def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Return log-probabilities after temperature, then top-k, then top-p. Removed tokens get -inf."""
    z = logits / temperature
    if top_k:
        kth = torch.topk(z, top_k).values[..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    if top_p < 1.0:
        srt, order = torch.sort(z, descending=True)
        p = srt.softmax(-1)
        drop = p.cumsum(-1) - p >= top_p          # mass before this token already reaches top_p
        z = z.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return z.log_softmax(-1)


@torch.no_grad()
def generate(model, ids, n, temperature=1.0, top_k=0, top_p=1.0, seed=0, cache=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids])
    c = model.new_cache(1, len(ids) + n) if cache else None
    logits = model(x, c)[:, -1]
    out = []
    for _ in range(n):
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(filter_logits(logits, temperature, top_k, top_p).exp(), 1, generator=g)
        out.append(nxt.item())
        x = torch.cat([x, nxt], 1)
        logits = model(nxt if cache else x, c)[:, -1]
    return out


def repeated_4gram_fraction(toks):
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1 - len(set(grams)) / len(grams)


def sampling(model, tok):
    ids = tok.encode(PROMPT).ids
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0, -1]
    rows = []
    for T in (0.5, 0.8, 1.0, 1.5):
        lp = filter_logits(logits, T)
        p = lp.exp()
        ent = -(p * lp.clamp_min(-1e9)).sum().item()
        nucleus = int((filter_logits(logits, T, top_p=0.9) > float("-inf")).sum())
        top = torch.topk(p, 3)
        rows.append({"temperature": T, "entropyNats": round(ent, 3), "top1Prob": round(top.values[0].item(), 4),
                     "top3": [[tok.decode([i]), round(v, 4)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
                     "nucleusSizeTopP0.9": nucleus,
                     "top50Mass": round(torch.topk(p, 50).values.sum().item(), 4)})
        print(rows[-1])
    samples, rep = [], {}
    for name, kw in [("greedy", dict(temperature=0)), ("T0.8-top50", dict(temperature=0.8, top_k=50)),
                     ("T1.0-topp0.9", dict(temperature=1.0, top_p=0.9)), ("T1.5", dict(temperature=1.5))]:
        fr = []
        for seed in range(5 if name != "greedy" else 1):
            out = generate(model, ids, 200, seed=seed, **kw)
            fr.append(repeated_4gram_fraction(out))
            if seed == 0:
                samples.append({"step": None, "prompt": PROMPT, "settings": name, "completion": tok.decode(out[:60])})
        rep[name] = {"repeated4gramFraction200tok": [round(f, 3) for f in fr], "mean": round(sum(fr) / len(fr), 3)}
        print(name, rep[name], samples[-1]["completion"][:200])
    write_run("lltGeneration-t1-sampling", "T1", "experiment:sampling", {"prompt": PROMPT, "promptTokens": len(ids),
              "vocab": logits.numel(), "newTokens": 200, "seeds": "0-4 for sampled settings"}, 0, [],
              evals={"nextToken": rows, "repetition": rep}, samples=samples)


@torch.no_grad()
def cache_exact(model, tok, run_id="lltGeneration-t1-cache-exact", n=200, label="baseline", bug=None, record=True):
    """Greedy decode with the cache, then recompute each step's logits with a full uncached forward."""
    ids = tok.encode(PROMPT).ids
    model.bug = bug
    c = model.new_cache(1, len(ids) + n)
    x = torch.tensor([ids])
    cached = [model(x, c)[0, -1]]
    toks = []
    for _ in range(n - 1):
        nxt = cached[-1].argmax().view(1, 1)
        toks.append(nxt.item())
        cached.append(model(nxt, c)[0, -1])
    toks.append(cached[-1].argmax().item())
    model.bug = None
    full = torch.tensor([ids + toks])
    ref = model(full[:, :-1])[0, len(ids) - 1:]
    cached = torch.stack(cached)
    err = (cached - ref).abs().amax(-1)
    ref_greedy = ref.argmax(-1).tolist()
    first_diff = next((i for i, (a, b) in enumerate(zip(toks, ref_greedy)) if a != b), None)
    # Teacher-forced: correct greedy continuation; NLL of the cached logits vs the uncached ones on the same tokens.
    res = {"promptTokens": len(ids), "steps": n, "maxAbsErr": float(err.max()), "maxAbsErrPrefillLast": float(err[0]),
           "maxAbsErrDecode": float(err[1:].max()), "meanAbsErrDecode": float((cached[1:] - ref[1:]).abs().mean()),
           "allclose_atol1e-4_rtol1e-4": bool(torch.allclose(cached, ref, atol=1e-4, rtol=1e-4)),
           "maxAbsLogit": float(ref.abs().max()),
           "firstStepWhereArgmaxDiffers": first_diff,
           "errAtStep": {str(k): float(err[k]) for k in (0, 1, 2, 5, 10, 50, 100, n - 1) if k < n}}
    print(run_id, res)
    if bug is None and record:
        write_run(run_id, "T1", label, {"prompt": PROMPT, "dtype": "float32", "decode": "greedy"}, 0, [], evals=res,
                  samples=[{"step": None, "prompt": PROMPT, "completion": tok.decode(toks[:60])}])
    return res, toks


@torch.no_grad()
def kv_bytes(model):
    c = model.cfg
    hd = c["d"] // c["H"]
    rows = []
    for T, dt, nb in [(256, torch.float32, 4), (256, torch.float16, 2), (1, torch.float32, 4)]:
        cache = model.new_cache(1, T, dt)
        formula = 2 * c["L"] * T * c["H_kv"] * hd * nb
        rows.append({"T": T, "dtype": str(dt), "formula": formula, "tensorBytes": cache.nbytes(),
                     "kShape": list(cache.k.shape)})
        print(rows[-1])
    # SmolLM2-135M through transformers: fill a real cache and add up its tensors.
    from transformers import AutoModelForCausalLM
    name = "HuggingFaceTB/SmolLM2-135M"
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    hc = hf.config
    T = 512
    x = torch.randint(0, hc.vocab_size, (1, T), generator=torch.Generator().manual_seed(0))
    pkv = hf(x, use_cache=True).past_key_values
    if hasattr(pkv, "layers"):
        tensors = [t for layer in pkv.layers for t in (layer.keys, layer.values)]
    else:
        tensors = [t for kv in pkv for t in kv]
    real = sum(t.nbytes for t in tensors)
    hd_hf = hc.hidden_size // hc.num_attention_heads
    smol = {"model": name, "layers": hc.num_hidden_layers, "kvHeads": hc.num_key_value_heads, "headDim": hd_hf, "T": T,
            "dtype": "float32", "formula": 2 * hc.num_hidden_layers * T * hc.num_key_value_heads * hd_hf * 4,
            "cacheTensorBytes": real, "keyShape": list(tensors[0].shape), "cacheClass": type(pkv).__name__,
            "perTokenBf16": 2 * hc.num_hidden_layers * hc.num_key_value_heads * hd_hf * 2}
    print(smol)
    tiny = {"model": "lltTinyStories baseline (d 320, 8 layers, 8 heads, head dim 40, ctx 512)",
            "perTokenFp16": 2 * 8 * 8 * 40 * 2, "fullContextFp16": 2 * 8 * 512 * 8 * 40 * 2}
    write_run("lltGeneration-t1-kv-bytes", "T1", "baseline", {"ours": model.cfg}, 0, [],
              evals={"ours": rows, "smollm2_135m": smol, "lltTinyStoriesDerived": tiny})


@torch.no_grad()
def gqa(tok):
    ids = tok.encode(PROMPT).ids
    rows = []
    for h_kv in (6, 3, 2, 1):
        torch.manual_seed(0)
        m = GPT(2048, 192, 4, 6, 256, h_kv).eval()
        cache = m.new_cache(1, 256)
        x = torch.tensor([ids])
        c = m.new_cache(1, len(ids) + 50)
        lg = [m(x, c)[0, -1]]
        toks = []
        for _ in range(49):
            nxt = lg[-1].argmax().view(1, 1)
            toks.append(nxt.item())
            lg.append(m(nxt, c)[0, -1])
        ref = m(torch.tensor([ids + toks]))[0, len(ids) - 1:]
        err = (torch.stack(lg) - ref).abs().max().item()
        rows.append({"kvHeads": h_kv, "groupSize": 6 // h_kv, "cacheBytesT256": cache.nbytes(),
                     "formula": 2 * 4 * 256 * h_kv * 32 * 4, "params": sum(p.numel() for p in m.parameters()),
                     "attnParamsPerLayer": m.blocks[0].qkv.weight.numel() + m.blocks[0].o.weight.numel(),
                     "maxAbsErrCachedVsUncached": err})
        print(rows[-1])
    write_run("lltGeneration-t1-gqa", "T1", "experiment:kv-heads", {"model": "GPT(2048, 192, 4 layers, 6 heads, ctx 256), random init seed 0",
              "kvHeads": [6, 3, 2, 1], "T": 256, "dtype": "float32"}, 0, [], evals={"rows": rows},
              notes="Random weights: this run measures cache size and cache exactness, not quality. GQA quality needs a trained model per setting.")


@torch.no_grad()
def broken(model, tok):
    ref_res, ref_toks = cache_exact(model, tok, n=60, label="control", record=False)
    # 1. RoPE offset forgotten.
    res, toks = cache_exact(model, tok, n=60, bug="rope_offset")
    write_run("lltGeneration-t1-broken-rope-offset", "T1", "broken:rope-offset", {"prompt": PROMPT, "bug": "RoPE positions start at 0 on every forward call, so every decoded token is rotated as position 0", "decode": "greedy"},
              0, [], evals=dict(res, correctFirst60=tok.decode(ref_toks)),
              samples=[{"step": None, "prompt": PROMPT, "settings": "correct cache", "completion": tok.decode(ref_toks)},
                       {"step": None, "prompt": PROMPT, "settings": "rope offset bug", "completion": tok.decode(toks)}])
    # Teacher-forced NLL of a held-out continuation through the buggy cache vs the correct one.
    text = PROMPT + " loved to play outside. One day, she saw a big red ball in the park. She ran to get it, but a dog took it first."
    tids = tok.encode(text).ids
    P = len(tok.encode(PROMPT).ids)
    out = {}
    for bug in (None, "rope_offset"):
        model.bug = bug
        c = model.new_cache(1, len(tids))
        lg = [model(torch.tensor([tids[:P]]), c)[0, -1]]
        for t in tids[P:-1]:
            lg.append(model(torch.tensor([[t]]), c)[0, -1])
        lg = torch.stack(lg)
        out[str(bug)] = round(F.cross_entropy(lg, torch.tensor(tids[P:])).item(), 4)
    model.bug = None
    print("teacher-forced NLL", out)
    rec_path = "lltGeneration-t1-broken-rope-offset"
    import json
    from runlog import RUNS_DIR
    with open(os.path.join(RUNS_DIR, rec_path + ".json"), encoding="utf-8") as f:
        rec = json.load(f)
    rec["evals"]["teacherForcedNll"] = {"text": text, "continuationTokens": len(tids) - P, "correct": out["None"], "ropeOffsetBug": out["rope_offset"]}
    with open(os.path.join(RUNS_DIR, rec_path + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)

    # 2. No causal mask in the prefill chunk.
    pids = tok.encode(text).ids
    x = torch.tensor([pids])
    y = torch.tensor(pids[1:])
    nll = {}
    for bug in (None, "prefill_mask"):
        model.bug = bug
        c = model.new_cache(1, len(pids))
        lg = model(x, c)[0, :-1]
        nll[str(bug)] = round(F.cross_entropy(lg, y).item(), 4)
    model.bug = None
    res2, toks2 = cache_exact(model, tok, n=60, bug="prefill_mask")
    print("prefill NLL", nll)
    write_run("lltGeneration-t1-broken-prefill-mask", "T1", "broken:prefill-mask",
              {"prompt": PROMPT, "bug": "the prefill chunk runs attention with no causal mask, so each prompt token reads later prompt tokens", "decode": "greedy"},
              0, [], evals=dict(res2, prefillNll={"text": text, "tokens": len(pids), "withMask": nll["None"], "withoutMask": nll["prefill_mask"]}),
              samples=[{"step": None, "prompt": PROMPT, "settings": "correct cache", "completion": tok.decode(ref_toks)},
                       {"step": None, "prompt": PROMPT, "settings": "prefill mask bug", "completion": tok.decode(toks2)}])


@torch.no_grad()
def gotchas(model, tok):
    ids = tok.encode(PROMPT).ids
    rec = {}
    # Cache in float16, model in float32.
    c = model.new_cache(1, len(ids) + 4, torch.float16)
    try:
        model(torch.tensor([ids]), c)
        rec["dtypeMismatch"] = "no error"
    except Exception as e:  # noqa: BLE001
        rec["dtypeMismatch"] = type(e).__name__ + ": " + str(e).splitlines()[0]
    # is_causal=True on a 1-token query against a longer key set: PyTorch aligns the mask top-left.
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 5, 8)
    v = torch.randn(1, 1, 5, 8)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rec["isCausalDecodeEqualsFirstValue"] = bool(torch.allclose(a[0, 0, 0], v[0, 0, 0], atol=1e-6))
    full = F.scaled_dot_product_attention(q, k, v)
    rec["isCausalDecodeMaxDiffVsNoMask"] = float((a - full).abs().max())
    # top_p=0.0 removes every token before multinomial.
    try:
        with torch.no_grad():
            lg = model(torch.tensor([ids]))[:, -1]
        torch.multinomial(filter_logits(lg, 1.0, 0, 0.0).exp(), 1)
        rec["topP0"] = "no error"
    except Exception as e:  # noqa: BLE001
        rec["topP0"] = type(e).__name__ + ": " + str(e).splitlines()[0]
    # A cache shorter than the prompt plus the generated tokens.
    c = model.new_cache(1, len(ids) + 2)
    try:
        with torch.no_grad():
            model(torch.tensor([ids]), c)
            for _ in range(4):
                model(torch.tensor([[5]]), c)
        rec["cacheOverflow"] = "no error"
    except Exception as e:  # noqa: BLE001
        rec["cacheOverflow"] = type(e).__name__ + ": " + str(e).splitlines()[0]
    print(rec)
    write_run("lltGeneration-t1-gotchas", "T1", "gotchas", {"torch": torch.__version__}, 0, [], evals=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from tokenizers import Tokenizer
    model, _ = load(os.path.join(OUT, "gen-model.pt"))
    tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    todo = args.only.split(",") if args.only else ["sampling", "exact", "bytes", "gqa", "broken", "gotchas"]
    if "sampling" in todo:
        sampling(model, tok)
    if "exact" in todo:
        cache_exact(model, tok)
    if "bytes" in todo:
        kv_bytes(model)
    if "gqa" in todo:
        gqa(tok)
    if "broken" in todo:
        broken(model, tok)
    if "gotchas" in todo:
        gotchas(model, tok)


if __name__ == "__main__":
    main()

"""lltIntro T1 measurements on a laptop CPU (no GPU).

1. Load SmolLM2-135M (pretrained) and the same architecture with random weights (3 seeds).
2. Loss on one English paragraph for both; compare with ln(V).
3. Generate from both (greedy, 30 new tokens).
4. English vs Python code vs Hindi: loss and tokens per word on the pretrained model.
5. Broken run: random init with initializer_range = 1.0 instead of the config's value.

Every number lands in a Run Library JSON under runs/lltIntro/.
Usage: python measure.py
"""
import math
import time

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from runlog import write_run
from texts import PROMPT, TEXTS

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROC = psutil.Process()


def peak_gb():
    mi = PROC.memory_info()
    peak = getattr(mi, "peak_wset", None) or mi.rss  # peak_wset exists on Windows
    return round(peak / 1e9, 2)


@torch.no_grad()
def text_loss(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    return model(ids, labels=ids).loss.item(), ids.shape[1]


@torch.no_grad()
def greedy(model, tok, prompt, n=30):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:])


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    V = len(tok)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    load_sec = time.time() - t_load
    n_params = sum(p.numel() for p in model.parameters())
    cfg = AutoConfig.from_pretrained(MODEL)
    print("V", V, "ln V", round(math.log(V), 4), "params", n_params, "load", round(load_sec, 1), "s")

    eng = TEXTS["English"]
    pre_loss, n_tok = text_loss(model, tok, eng)
    pre_gen = greedy(model, tok, PROMPT)
    print("pretrained loss", round(pre_loss, 4), "on", n_tok, "tokens")
    print("pretrained:", repr(pre_gen))

    lang = {}
    for name, text in TEXTS.items():
        loss, nt = text_loss(model, tok, text)
        words = len(text.split())
        lang[name] = {"loss": round(loss, 4), "tokens": nt, "words": words,
                      "tokensPerWord": round(nt / words, 2), "bytes": len(text.encode("utf-8")),
                      "bytesPerToken": round(len(text.encode("utf-8")) / nt, 2)}
        print(name, lang[name])

    rand = []
    rand_gen = None
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        rm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
        rl, _ = text_loss(rm, tok, eng)
        rand.append(round(rl, 4))
        if seed == 0:
            rand_gen = greedy(rm, tok, PROMPT)
        del rm
    print("random-init loss by seed", rand, "| ln V", round(math.log(V), 4))
    print("random:", repr(rand_gen))
    wall = time.time() - t0

    common = {"model": MODEL, "dtype": "float32", "threads": 4, "vocab_size": V,
              "initializer_range": cfg.initializer_range, "tie_word_embeddings": cfg.tie_word_embeddings,
              "n_params": n_params}
    write_run("lltIntro-t1-smollm2-135m", "T1", "baseline", common, 0,
              metrics=[{"step": 0, "pretrainedLossEnglish": round(pre_loss, 4), "englishTokens": n_tok,
                        "randomInitLossEnglish": rand, "lnV": round(math.log(V), 4)}],
              evals={"byLanguage": lang, "loadSec": round(load_sec, 1), "peakRamGB": peak_gb()},
              samples=[{"step": 0, "prompt": PROMPT, "model": "pretrained", "decoding": "greedy, 30 new tokens",
                        "completion": pre_gen},
                       {"step": 0, "prompt": PROMPT, "model": "random init, seed 0",
                        "decoding": "greedy, 30 new tokens", "completion": rand_gen}],
              wall_clock_sec=round(wall, 1))

    # Broken: the same architecture initialised with std 1.0 instead of the config's value.
    bad = []
    for seed in (0, 1, 2):
        c2 = AutoConfig.from_pretrained(MODEL)
        c2.initializer_range = 1.0
        torch.manual_seed(seed)
        bm = AutoModelForCausalLM.from_config(c2, dtype=torch.float32).eval()
        bl, _ = text_loss(bm, tok, eng)
        bad.append(round(bl, 4))
        del bm
    print("init std 1.0 loss by seed", bad)
    write_run("lltIntro-t1-broken-init-range-1", "T1", "broken:initializer_range=1.0",
              dict(common, initializer_range=1.0), 0,
              metrics=[{"step": 0, "randomInitLossEnglish": bad, "lnV": round(math.log(V), 4)}],
              evals={"seeds": [0, 1, 2], "peakRamGB": peak_gb()},
              wall_clock_sec=round(time.time() - t0, 1))
    print("total wall", round(time.time() - t0, 1), "s | peak RAM", peak_gb(), "GB")


if __name__ == "__main__":
    main()

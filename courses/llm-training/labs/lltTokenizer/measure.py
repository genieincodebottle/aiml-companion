"""Measure every figure the lltTokenizer lab page quotes, and write Run Library JSON.

Run `python data.py` first. Then:
  python measure.py all          (every experiment, CPU only)
  python measure.py sweep        (one experiment; names below)

Experiments and run ids (files in ../../runs/lltTokenizer/):
  pieces    lltTokenizer-t1-pretokenize  pieces per word under GPT-2's rule and the vowel-sign rule
  inpage    lltTokenizer-t0-inpage       the browser steps' corpus, timed in CPython
  parity    lltTokenizer-t1-parity       our encoder with GPT-2's merges vs Hugging Face, plus speed
  trainspeed lltTokenizer-t1-trainspeed  our trainer vs Hugging Face's trainer on the same text
  sweep     lltTokenizer-t1-vocab-sweep  vocab size 512 to 65,536, bytes per token per language
  regex     lltTokenizer-t1-regex-off-8k    broken run: no pre-tokenization
  digits    lltTokenizer-t1-digits       digit runs vs single digits
  fertility lltTokenizer-t1-fertility    tokens per word: GPT-2, SmolLM2, Qwen3 and ours
  nfkc      lltTokenizer-t1-nfkc         broken run: an NFKC normalizer changes the text
  mismatch  lltTokenizer-t1-gpt2-mismatch     broken run: GPT-2 fed SmolLM2 token ids
"""
import json
import math
import os
import platform
import random
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpe  # noqa: E402
import data  # noqa: E402
from data import heldout, lines  # noqa: E402

CACHE = Path(os.environ.get("LLT_TOK_CACHE", Path.home() / ".cache" / "lltlab" / "tokenizer"))
RUNS = HERE.parent.parent / "runs" / "lltTokenizer"
LANGS = ("en", "hi", "es")
SEED = 0

# The browser steps' corpus, identical to browser/step2.py. Written for this lab.
INPAGE_CORPUS = """The tokenizer reads text as bytes. It counts every pair of neighbouring bytes and merges the most frequent pair into a new token. Then it counts again. After a few hundred merges, common words such as the, and, token and text are single tokens.
टोकनाइज़र टेक्स्ट को बाइट्स के रूप में पढ़ता है। वह पास-पास के हर जोड़े को गिनता है और सबसे ज़्यादा आने वाले जोड़े को एक नए टोकन में जोड़ देता है। फिर वह दोबारा गिनता है। कुछ सौ मर्ज के बाद आम शब्द एक-एक टोकन बन जाते हैं।
El tokenizador lee el texto como bytes. Cuenta cada par de bytes vecinos y une el par más frecuente en un token nuevo. Después vuelve a contar. Tras unos cientos de fusiones, las palabras comunes como el, de, token y texto son un solo token."""


# ---------------------------------------------------------------- helpers
def env():
    import numpy, tokenizers, transformers, huggingface_hub
    out = {"python": platform.python_version(), "tokenizers": tokenizers.__version__,
           "transformers": transformers.__version__, "numpy": numpy.__version__,
           "huggingface_hub": huggingface_hub.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import datasets
        out["datasets"] = datasets.__version__
    except ImportError:
        pass
    return out


def hardware():
    return {"gpu": None, "count": 0, "vramGB": 0, "cpu": cpu_name(),
            "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
            "ramGB": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform(),
           "cpuBusyPctAtEnd": psutil.cpu_percent(interval=1)}


def cpu_name():
    if platform.system() == "Windows":
        try:
            import subprocess
            return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return platform.processor()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def peak_mem_gb():
    p = psutil.Process()
    info = p.memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return round(peak / 1e9, 3)


def write_run(run_id, tier, label, config, metrics, evals, wall, samples=None):
    RUNS.mkdir(parents=True, exist_ok=True)
    rec = {"runId": run_id, "module": "lltTokenizer", "tier": tier, "label": label,
           "hardware": hardware(), "env": env(), "notebookCommit": None,
           "config": config, "seed": SEED, "metrics": metrics, "evals": evals,
           "samples": samples or [], "wallClockSec": round(wall, 2), "peakMemGB": peak_mem_gb(),
           "costUSD": 0, "priceCheckedOn": None, "date": date.today().isoformat()}
    (RUNS / f"{run_id}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", run_id, f"{wall:.1f}s")
    return rec


def train_text():
    return data.train_texts()


DATA = {
    "heldout": {
        "flores": "FLORES-200 devtest (eng_Latn, hin_Deva, spa_Latn), 1,012 parallel sentences each, CC BY-SA 4.0, "
                  "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz (sha256 b8b0b767...011f6)",
        "code": "12 CPython 3.12.3 standard library files (every 10th file, see data.py), PSF License",
    },
    "train": {
        "wikipedia": "wikimedia/wikipedia 20231101.en, .hi, .es: first articles in file order up to 8 MB each, "
                     "CC BY-SA 3.0 and GFDL",
        "code": "CPython 3.12.3 standard library files not held out, up to 4 MB, PSF License",
    },
    "words": "tokens per word = tokens / number of whitespace-separated words (str.split())",
}


def stats(n_tokens, text):
    nb = len(text.encode("utf-8"))
    nw = len(text.split())
    return {"tokens": n_tokens, "bytes": nb, "words": nw,
            "bytesPerToken": round(nb / n_tokens, 3), "tokensPerWord": round(n_tokens / nw, 3)}


def hf_bpe(vocab_size, texts, pre="bytelevel", normalizer=None):
    """Train a byte-level BPE with Hugging Face tokenizers. pre: bytelevel | noregex | digits."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
    tok = Tokenizer(models.BPE())
    if pre == "bytelevel":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    elif pre == "noregex":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    elif pre == "digits":
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    if normalizer == "nfkc":
        tok.normalizer = normalizers.NFKC()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.perf_counter()
    tok.train_from_iterator(texts, trainer=trainer)
    return tok, time.perf_counter() - t0


def hf_load(repo):
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def count(tok, text):
    return sum(len(e.ids) for e in tok.encode_batch(text.splitlines(), add_special_tokens=False))


# ---------------------------------------------------------------- experiments
def exp_inpage():
    """Run browser/step1.py to step4.py (the page's in-browser cells) in CPython and record
    their output, then train on the same corpus with pre-tokenization for comparison."""
    import subprocess
    t0 = time.perf_counter()
    steps = {}
    for n in (1, 2, 3, 4):
        path = HERE / "browser" / f"step{n}.py"
        s0 = time.perf_counter()
        out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, encoding="utf-8",
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"), check=True).stdout
        steps[f"step{n}"] = {"stdout": out, "processSec": round(time.perf_counter() - s0, 3)}
    corpus = INPAGE_CORPUS
    m_raw = bpe.train(corpus, 256 + 100, use_pretokenize=False)
    m_word = bpe.train(corpus, 256 + 100)
    v_raw, v_word = bpe.build_vocab(m_raw), bpe.build_vocab(m_word)
    cross = lambda m, v: sum(b" " in v[i].lstrip(b" ") for i in m.values())
    ids = bpe.encode(corpus, m_word)
    assert bpe.decode(ids, v_word) == corpus
    evals = {"steps": steps, "corpusBytes": len(corpus.encode("utf-8")),
             "raw100": {"crossBoundary": cross(m_raw, v_raw)},
             "pieces100": {"crossBoundary": cross(m_word, v_word), "tokens": len(ids)}}
    write_run("lltTokenizer-t0-inpage", "T0", "baseline",
              {"corpus": "the three-line corpus in browser/step2.py, written for this lab (English, Hindi, Spanish)",
               "merges": 100, "runtime": "CPython stand-in for Pyodide; see lltTokenizer-t0-pyodide for Pyodide 0.27.0"},
              [], evals, time.perf_counter() - t0)


def exp_parity():
    from huggingface_hub import hf_hub_download
    t0 = time.perf_counter()
    path = hf_hub_download("gpt2", "tokenizer.json")
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    ml = [m.split(" ") if isinstance(m, str) else m for m in spec["model"]["merges"]]
    merges, vocab = bpe.merges_from_hf(ml)
    hf = hf_load("gpt2")
    hf_vocab = {i: s for s, i in spec["model"]["vocab"].items()}
    byte_of = {c: b for b, c in bpe.bytes_to_unicode().items()}
    hfbytes = lambda i: bytes(byte_of[c] for c in hf_vocab[i])
    ho = heldout()
    texts = data.heldout_texts()
    same = sum([vocab[i] for i in bpe.encode(t, merges)] == [hfbytes(i) for i in hf.encode(t).ids] for t in texts)
    # pre-tokenization on random Unicode vs the library's regex
    rng = random.Random(SEED)
    rand = []
    for _ in range(5000):
        s = []
        for _ in range(rng.randrange(1, 40)):
            while True:
                cp = rng.randrange(0x110000) if rng.random() < 0.5 else rng.randrange(0x3000)
                if not 0xD800 <= cp <= 0xDFFF:
                    break
            s.append(chr(cp))
        rand.append("".join(s))
    pre_diff = enc_diff = 0
    for s in rand:
        mine = bpe.pretokenize(s)
        lib = [s[a:b] for _, (a, b) in hf.pre_tokenizer.pre_tokenize_str(s)]
        if mine != lib:
            pre_diff += 1
        if [vocab[i] for i in bpe.encode(s, merges)] != [hfbytes(i) for i in hf.encode(s).ids]:
            enc_diff += 1
    # which characters disagree: those Python's Unicode tables call unassigned (Cn) or newer
    # every code point, one at a time between two letters: where do the two splitters disagree?
    diff_cats = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        s = "a" + chr(cp) + "a"
        if bpe.pretokenize(s) != [s[a:b] for _, (a, b) in hf.pre_tokenizer.pre_tokenize_str(s)]:
            cat = unicodedata.category(chr(cp))
            diff_cats[cat] = diff_cats.get(cat, 0) + 1
    big = "\n".join(ho[k] for k in LANGS)
    nbytes = len(big.encode("utf-8"))
    big_lines = big.splitlines()
    t = time.perf_counter(); mine_n = sum(len(bpe.encode(x, merges)) for x in big_lines); t_mine_nocache = time.perf_counter() - t
    cache = {}
    t = time.perf_counter(); mine_n2 = sum(len(bpe.encode(x, merges, cache=cache)) for x in big_lines); t_mine = time.perf_counter() - t
    t = time.perf_counter(); hf_n = sum(len(hf.encode(x).ids) for x in big_lines); t_hf1 = time.perf_counter() - t
    t = time.perf_counter(); hf_nb = sum(len(e.ids) for e in hf.encode_batch(big_lines)); t_hfb = time.perf_counter() - t
    # token-by-token decode: how many GPT-2 tokens are not valid UTF-8 on their own
    partial = {}
    for k in LANGS:
        ids = [i for x in ho[k].splitlines() for i in hf.encode(x).ids]
        bad = 0
        for i in ids:
            try:
                hfbytes(i).decode("utf-8")
            except UnicodeDecodeError:
                bad += 1
        partial[k] = {"tokens": len(ids), "invalidAlone": bad, "share": round(bad / len(ids), 4)}
    evals = {"texts": len(texts), "identical": same, "random": {"strings": len(rand), "preTokenizeDiffer": pre_diff,
             "encodingDiffer": enc_diff, "codePointsThatDiffer": diff_cats, "pythonUnicode": unicodedata.unidata_version},
             "speed": {"bytes": nbytes, "tokens": mine_n, "tokensHF": hf_n, "tokensHFBatch": hf_nb,
                       "oursNoCacheSec": round(t_mine_nocache, 3), "oursCacheSec": round(t_mine, 3),
                       "hfLoopSec": round(t_hf1, 3), "hfBatchSec": round(t_hfb, 3),
                       "oursCacheMBps": round(nbytes / t_mine / 1e6, 3), "hfBatchMBps": round(nbytes / t_hfb / 1e6, 2),
                       "speedupBatchVsOurs": round(t_mine / t_hfb, 1), "speedupLoopVsOurs": round(t_mine / t_hf1, 1)},
             "tokenByTokenDecode": partial}
    assert mine_n == mine_n2 == hf_n == hf_nb
    write_run("lltTokenizer-t1-parity", "T1", "baseline",
              {"merges": "GPT-2 tokenizer.json from the Hugging Face repo gpt2 (50,000 merges)", "data": DATA["heldout"]},
              [], evals, time.perf_counter() - t0)


def exp_trainspeed():
    t0 = time.perf_counter()
    text = data.train_slice(250_000)  # 250 KB from each source, 1 MB
    V = 1024
    t = time.perf_counter(); m = bpe.train(text, V); t_ours = time.perf_counter() - t
    vocab = bpe.build_vocab(m)
    hf, t_hf = hf_bpe(V, text.splitlines())
    ho = heldout()
    comp = {}
    for k, v in ho.items():
        cache = {}
        n_ours = sum(len(bpe.encode(x, m, cache=cache)) for x in v.splitlines())
        comp[k] = {"ours": stats(n_ours, v)["bytesPerToken"], "hf": stats(count(hf, v), v)["bytesPerToken"]}
    # how many of our merges produce a token string the library also learned
    ours_tokens = {vocab[i] for i in m.values()}
    byte_of = {c: b for b, c in bpe.bytes_to_unicode().items()}
    hf_tokens = {bytes(byte_of[c] for c in s) for s, i in hf.get_vocab().items() if i >= 256}
    evals = {"textBytes": len(text.encode("utf-8")), "vocabSize": V, "oursSec": round(t_ours, 2), "hfSec": round(t_hf, 3),
             "speedup": round(t_ours / t_hf), "heldoutBytesPerToken": comp,
             "sharedTokens": len(ours_tokens & hf_tokens), "oursTokens": len(ours_tokens), "hfTokens": len(hf_tokens)}
    write_run("lltTokenizer-t1-trainspeed", "T1", "baseline",
              {"data": DATA, "slice": "first 250,000 bytes of each training source (1 MB total)", "vocabSize": V},
              [], evals, time.perf_counter() - t0)


SWEEP = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def exp_sweep():
    t0 = time.perf_counter()
    tt = train_text()
    ho = heldout()
    metrics = []
    for V in SWEEP:
        tok, sec = hf_bpe(V, lines(tt.values()))
        row = {"step": V, "vocabSize": V, "trainSec": round(sec, 2)}
        for k, v in ho.items():
            s = stats(count(tok, v), v)
            row[f"{k}BytesPerToken"] = s["bytesPerToken"]
            row[f"{k}TokensPerWord"] = s["tokensPerWord"]
        metrics.append(row)
        print(row)
        if V == 32768:
            tok.save(str(CACHE / "ours_32k.json"))
    write_run("lltTokenizer-t1-vocab-sweep", "T1", "ablation:vocabSize",
              {"data": DATA, "trainer": "tokenizers BpeTrainer, ByteLevel pre-tokenizer (GPT-2 regex)", "sizes": list(SWEEP)},
              metrics, {}, time.perf_counter() - t0)


def cross_boundary(tok):
    """Learned tokens that hold a space after a non-space byte, so they cross a word boundary."""
    byte_of = {c: b for b, c in bpe.bytes_to_unicode().items()}
    toks = [bytes(byte_of[c] for c in s) for s, i in tok.get_vocab().items() if i >= 256]
    cross = [t for t in toks if b" " in t.lstrip(b" ")]
    return len(toks), cross


def exp_regex():
    t0 = time.perf_counter()
    tt = train_text()
    ho = heldout()
    V = 8192
    out = {}
    for name, pre in (("regexOn", "bytelevel"), ("regexOff", "noregex")):
        tok, sec = hf_bpe(V, lines(tt.values()), pre=pre)
        n, cross = cross_boundary(tok)
        held = {}
        for k, v in ho.items():
            enc = tok.encode_batch(v.splitlines(), add_special_tokens=False)
            ntok = sum(len(e.ids) for e in enc)
            inside = sum(1 for e in enc for t in e.tokens if "Ġ" in t.lstrip("Ġ"))
            held[k] = stats(ntok, v) | {"tokensSpanningASpace": inside, "shareSpanning": round(inside / ntok, 4)}
        out[name] = {"trainSec": round(sec, 2), "learned": n, "crossBoundary": len(cross),
                     "crossShare": round(len(cross) / n, 4), "heldout": held}
        if name == "regexOff":
            ranked = sorted(((i, s) for s, i in tok.get_vocab().items() if i >= 256), key=lambda x: x[0])
            byte_of = {c: b for b, c in bpe.bytes_to_unicode().items()}
            firsts = []
            for i, s in ranked:
                b = bytes(byte_of[c] for c in s)
                if b" " in b.lstrip(b" ") and b.isascii():
                    firsts.append(b.decode())
                if len(firsts) == 15:
                    break
            out[name]["firstCrossTokens"] = firsts
    write_run("lltTokenizer-t1-regex-off-8k", "T1", "broken:no-pretokenize",
              {"data": DATA, "vocabSize": V, "compare": "ByteLevel(use_regex=True) vs ByteLevel(use_regex=False)"},
              [], out, time.perf_counter() - t0)


def exp_digits():
    t0 = time.perf_counter()
    tt = train_text()
    ho = heldout()
    V = 32768
    out = {}
    nums = [f" {n}" for n in range(10000)]
    for name, pre in (("digitRuns", "bytelevel"), ("singleDigits", "digits")):
        tok, sec = hf_bpe(V, lines(tt.values()), pre=pre)
        lens = [len(e.ids) for e in tok.encode_batch(nums, add_special_tokens=False)]
        digit_tokens = sum(1 for s, i in tok.get_vocab().items() if i >= 256 and s.lstrip("Ġ").isdigit())
        # segmentation patterns of four-digit numbers, e.g. [2, 2] or [1, 3]
        pats = {}
        for n, e in zip(range(1000, 10000), tok.encode_batch(nums[1000:], add_special_tokens=False)):
            key = tuple(len(t.lstrip("Ġ")) for t in e.tokens)
            pats[key] = pats.get(key, 0) + 1
        held = {k: stats(count(tok, v), v)["bytesPerToken"] for k, v in ho.items()}
        out[name] = {"trainSec": round(sec, 2), "meanTokensPerNumber0to9999": round(sum(lens) / len(lens), 3),
                     "fourDigitPatterns": len(pats), "topPatterns": sorted(([list(k), v] for k, v in pats.items()), key=lambda x: -x[1])[:5],
                     "digitOnlyVocab": digit_tokens, "heldoutBytesPerToken": held}
    g = hf_load("gpt2")
    lens = [len(e.ids) for e in g.encode_batch(nums, add_special_tokens=False)]
    pats = {}
    for e in g.encode_batch(nums[1000:], add_special_tokens=False):
        key = tuple(len(t.lstrip("Ġ")) for t in e.tokens)
        pats[key] = pats.get(key, 0) + 1
    out["gpt2"] = {"meanTokensPerNumber0to9999": round(sum(lens) / len(lens), 3), "fourDigitPatterns": len(pats),
                   "singleTokenNumbers": sum(1 for x in lens if x == 1),
                   "topPatterns": sorted(([list(k), v] for k, v in pats.items()), key=lambda x: -x[1])[:5]}
    write_run("lltTokenizer-t1-digits", "T1", "ablation:digits",
              {"data": DATA, "vocabSize": V, "numbers": "' 0' to ' 9999', each with a leading space"},
              [], out, time.perf_counter() - t0)


def hf_bpe_pieces(vocab_size, pieces):
    """Train on pieces that are already split (our pretokenize), one piece per item."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.perf_counter()
    tok.train_from_iterator(pieces, trainer=trainer)
    return tok, time.perf_counter() - t0


def line_pieces(text, marks=False):
    """Pre-tokenize each line on its own, the same basis as count(): no newline pieces."""
    return [p for x in text.splitlines() for p in bpe.pretokenize(x, marks=marks)]


def count_marks(tok, text):
    return sum(len(e.ids) for e in tok.encode_batch(line_pieces(text, marks=True), add_special_tokens=False))


def exp_fertility():
    from tokenizers import Tokenizer
    t0 = time.perf_counter()
    ho = heldout()
    tt = train_text()
    toks = {"gpt2": hf_load("gpt2"), "smollm2": hf_load("HuggingFaceTB/SmolLM2-135M"), "qwen3": hf_load("Qwen/Qwen3-0.6B")}
    ours_path = CACHE / "ours_32k.json"
    if not ours_path.exists():
        tok, _ = hf_bpe(32768, lines(tt.values()))
        tok.save(str(ours_path))
    toks["ours32k"] = Tokenizer.from_file(str(ours_path))
    out = {}
    for name, tok in toks.items():
        out[name] = {"vocabSize": tok.get_vocab_size()}
        for k, v in ho.items():
            out[name][k] = stats(count(tok, v), v)
    # variants: vowel signs count as letters (marks=True); and an English+Hindi only mix
    variants = {"ours32kMarks": ("en", "hi", "es", "code"), "enhi32k": ("en", "hi"), "enhi32kMarks": ("en", "hi")}
    for name, srcs in variants.items():
        marks = name.endswith("Marks")
        text = "\n".join(tt[k] for k in srcs)
        if marks:
            tok, sec = hf_bpe_pieces(32768, bpe.pretokenize(text, marks=True))
        else:
            tok, sec = hf_bpe(32768, lines(tt[k] for k in srcs))
        out[name] = {"vocabSize": tok.get_vocab_size(), "trainSec": round(sec, 2), "sources": list(srcs)}
        for k, v in ho.items():
            out[name][k] = stats(count_marks(tok, v) if marks else count(tok, v), v)
    for name in out:
        out[name]["hiOverEnFertility"] = round(out[name]["hi"]["tokensPerWord"] / out[name]["en"]["tokensPerWord"], 2)
    write_run("lltTokenizer-t1-fertility", "T1", "baseline",
              {"data": DATA, "tokenizers": {"gpt2": "gpt2", "smollm2": "HuggingFaceTB/SmolLM2-135M",
                                            "qwen3": "Qwen/Qwen3-0.6B", "ours32k": "lltTokenizer-t1-vocab-sweep at 32,768",
                                            "ours32kMarks": "same mix, vowel signs count as letters (bpe.pretokenize marks=True), held-out text split per line",
                                            "enhi32k": "English and Hindi Wikipedia only, GPT-2 rule",
                                            "enhi32kMarks": "English and Hindi Wikipedia only, marks=True"}},
              [], out, time.perf_counter() - t0)


def exp_nfkc():
    t0 = time.perf_counter()
    ho = heldout()
    out = {}
    for k in LANGS:
        sents = [s for s in ho[k].splitlines() if s]
        out[k] = {"sentences": len(sents),
                  "changedByNFC": sum(unicodedata.normalize("NFC", s) != s for s in sents),
                  "changedByNFKC": sum(unicodedata.normalize("NFKC", s) != s for s in sents)}
    code = [c for c in ho["code"].split("\n\n") if c]
    out["code"] = {"chunks": len(code), "changedByNFKC": sum(unicodedata.normalize("NFKC", s) != s for s in code)}
    # which characters change in the Hindi sentences
    changed = {}
    for s in ho["hi"].splitlines():
        for ch in s:
            if unicodedata.normalize("NFKC", ch) != ch:
                key = f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"
                changed[key] = changed.get(key, 0) + 1
    out["hiChangedChars"] = sorted(changed.items(), key=lambda x: -x[1])[:10]
    # a byte-level tokenizer with an NFKC normalizer: decode(encode(s)) == s ?
    tok, _ = hf_bpe(4096, lines(v for v in train_text().values()), normalizer="nfkc")
    rt = {}
    for k in LANGS:
        sents = [s for s in ho[k].splitlines() if s]
        rt[k] = sum(tok.decode(tok.encode(s).ids) != s for s in sents)
    out["roundTripFailuresWithNFKC"] = rt
    # shipped tokenizers: which ones normalize, and does decode(encode(s)) give s back?
    shipped = {}
    for name, repo in (("gpt2", "gpt2"), ("smollm2", "HuggingFaceTB/SmolLM2-135M"), ("qwen3", "Qwen/Qwen3-0.6B")):
        t = hf_load(repo)
        shipped[name] = {"normalizer": type(t.normalizer).__name__ if t.normalizer else None}
        for k in LANGS:
            sents = [s for s in ho[k].splitlines() if s]
            shipped[name][k] = sum(t.decode(t.encode(s).ids) != s for s in sents)
    out["shippedRoundTripFailures"] = shipped
    samples = [{"step": 0, "prompt": "x² ﬁle １２３", "completion": tok.decode(tok.encode("x² ﬁle １２３").ids)}]
    write_run("lltTokenizer-t1-nfkc", "T1", "broken:nfkc-normalizer",
              {"data": DATA["heldout"], "vocabSize": 4096}, [], out, time.perf_counter() - t0, samples)


def exp_mismatch():
    import torch
    from transformers import AutoModelForCausalLM
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    sents = [s for s in heldout()["en"].splitlines() if s][:200]
    toks = {"gpt2": hf_load("gpt2"), "smollm2": hf_load("HuggingFaceTB/SmolLM2-135M"), "qwen3": hf_load("Qwen/Qwen3-0.6B")}
    out = {"lnV": round(math.log(50257), 3)}
    for name, tok in toks.items():
        nll, n, too_big = 0.0, 0, 0
        for s in sents:
            ids = tok.encode(s, add_special_tokens=False).ids
            if max(ids) >= model.config.vocab_size:
                too_big += 1
                continue
            x = torch.tensor([ids])
            with torch.no_grad():
                loss = model(x, labels=x).loss.item()
            nll += loss * (len(ids) - 1)
            n += len(ids) - 1
        out[name] = {"sentences": len(sents) - too_big, "skippedIndexError": too_big, "predictedTokens": n,
                     "meanLossNats": round(nll / n, 3) if n else None}
    # what a Qwen3 id above GPT-2's last row does when it reaches the model
    big = next(s for s in sents if max(toks["qwen3"].encode(s, add_special_tokens=False).ids) >= model.config.vocab_size)
    try:
        with torch.no_grad():
            model(torch.tensor([toks["qwen3"].encode(big, add_special_tokens=False).ids]))
        out["qwen3"]["error"] = None
    except Exception as e:
        out["qwen3"]["error"] = f"{type(e).__name__}: {e}"
    # what the learner sees: decode the mismatched ids with the model's own tokenizer
    s = sents[0]
    wrong = toks["smollm2"].encode(s, add_special_tokens=False).ids
    samples = [{"step": 0, "prompt": s, "completion": toks["gpt2"].decode(wrong)}]
    write_run("lltTokenizer-t1-gpt2-mismatch", "T1", "broken:tokenizer-model-mismatch",
              {"model": "gpt2 (124M), torch on 4 CPU threads", "text": "first 200 FLORES-200 devtest English sentences, CC BY-SA 4.0",
               "loss": "mean next-token cross-entropy per predicted token, no BOS"},
              [], out, time.perf_counter() - t0, samples)


def exp_pieces():
    """Pre-tokenization pieces per whitespace word, GPT-2's rule vs vowel signs counted as letters."""
    t0 = time.perf_counter()
    ho = heldout()
    out = {}
    for k in LANGS:
        words = ho[k].split()
        out[k] = {"words": len(words),
                  "wordsCutGpt2Rule": sum(len(bpe.pretokenize(w)) > 1 for w in words),
                  "pieces": len(line_pieces(ho[k])),
                  "piecesMarksRule": len(line_pieces(ho[k], marks=True)),
                  "piecesPerWordGpt2Rule": round(len(line_pieces(ho[k])) / len(words), 3),
                  "piecesPerWordMarksRule": round(len(line_pieces(ho[k], marks=True)) / len(words), 3)}
        out[k]["shareCutGpt2Rule"] = round(out[k]["wordsCutGpt2Rule"] / len(words), 3)
    gpt2 = hf_load("gpt2")
    cross = [t for t, i in gpt2.get_vocab().items() if i >= 256 and "Ġ" in t.lstrip("Ġ")]
    out["gpt2CrossingTokens"] = [[gpt2.token_to_id(t), " ".join(f"U+{b:04X}" for b in bytes(
        {c: b for b, c in bpe.bytes_to_unicode().items()}[ch] for ch in t))] for t in cross]
    write_run("lltTokenizer-t1-pretokenize", "T1", "baseline",
              {"data": DATA["heldout"], "rules": "bpe.pretokenize (GPT-2) vs bpe.pretokenize(marks=True)",
               "basis": "pieces counted per line (text.splitlines()), the same basis as every token count"},
              [], out, time.perf_counter() - t0)


EXPS = {"pieces": exp_pieces, "inpage": exp_inpage, "parity": exp_parity, "trainspeed": exp_trainspeed, "sweep": exp_sweep,
        "regex": exp_regex, "digits": exp_digits, "fertility": exp_fertility, "nfkc": exp_nfkc, "mismatch": exp_mismatch}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for name in (EXPS if names == ["all"] else names):
        EXPS[name]()

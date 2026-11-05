"""Byte-level BPE tokenizer in pure Python (LLM Training Lab, lltTokenizer).

The same functions the lab page builds step by step, collected in one file so the
notebook and measure.py import one copy. No dependencies beyond the standard library,
so it also runs in Pyodide.

Pre-tokenization follows GPT-2's rules without the `regex` module:
  's 't 're 've 'm 'll 'd | optional space + letters | optional space + numbers |
  optional space + other symbols | whitespace not followed by a non-space | whitespace
Letters are str.isalpha() (Unicode category L), numbers str.isnumeric(), whitespace
str.isspace(). measure.py checks the result against Hugging Face's GPT-2 tokenizer.
"""
import unicodedata
from collections import Counter
from itertools import groupby

CONTRACTIONS = ("s", "t", "re", "ve", "m", "ll", "d")


def kind(ch, marks=False):
    if ch.isalpha() or (marks and unicodedata.category(ch)[0] == "M"):
        return "L"
    if ch.isnumeric():
        return "N"
    if ch.isspace():
        return "S"
    return "P"


def pretokenize(text, digits="runs", marks=False):
    """Split text into pieces. digits='runs' keeps digit runs whole (GPT-2);
    digits='single' splits every digit into its own piece. marks=True counts combining
    marks (Devanagari vowel signs and virama, category M) as letters, so a Hindi word
    stays one piece; GPT-2's rule (marks=False) cuts it at every vowel sign."""
    key = (lambda ch: kind(ch, True)) if marks else kind
    runs = [[k, "".join(g)] for k, g in groupby(text, key)]
    pieces, lead = [], ""
    for i, (k, run) in enumerate(runs):
        if not run:
            continue
        last = i == len(runs) - 1
        if k == "S":
            if last:
                pieces.append(run)
                continue
            if len(run) > 1:
                pieces.append(run[:-1])
            if run[-1] == " ":
                lead = " "
            else:
                pieces.append(run[-1])
            continue
        if run == "'" and not lead and not last and runs[i + 1][0] == "L":
            nxt = runs[i + 1][1]
            c = next((c for c in CONTRACTIONS if nxt.startswith(c)), None)
            if c:
                pieces.append("'" + c)
                runs[i + 1][1] = nxt[len(c):]
                continue
        if k == "N" and digits == "single":
            pieces.append(lead + run[0])
            pieces.extend(run[1:])
        else:
            pieces.append(lead + run)
        lead = ""
    return pieces


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, vocab_size, use_pretokenize=True, digits="runs", verbose=False):
    """Learn merges. Returns {pair: new_id}; new_id order is the merge rank."""
    pieces = pretokenize(text, digits) if use_pretokenize else [text]
    words = Counter(tuple(p.encode("utf-8")) for p in pieces)
    merges = {}
    for new_id in range(256, vocab_size):
        stats = Counter()
        for w, n in words.items():
            for pair in zip(w, w[1:]):
                stats[pair] += n
        if not stats:
            break
        pair = max(stats, key=stats.get)
        merges[pair] = new_id
        words = {tuple(merge(list(w), pair, new_id)): n for w, n in words.items()}
        if verbose and new_id % 500 == 0:
            print(new_id, stats[pair])
    return merges


def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), i in merges.items():
        vocab[i] = vocab[a] + vocab[b]
    return vocab


def encode_piece(b, merges):
    ids = list(b)
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def encode(text, merges, digits="runs", cache=None):
    out = []
    for p in pretokenize(text, digits):
        if cache is not None:
            ids = cache.get(p)
            if ids is None:
                ids = cache[p] = encode_piece(p.encode("utf-8"), merges)
        else:
            ids = encode_piece(p.encode("utf-8"), merges)
        out.extend(ids)
    return out


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8")


def bytes_to_unicode():
    """GPT-2's map from bytes to printable characters, used in its merges file."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def merges_from_hf(merge_list):
    """Turn a Hugging Face merges list (pairs of byte-mapped strings, rank order)
    into this file's {pair: new_id} form. Returns (merges, vocab)."""
    byte_of = {c: b for b, c in bytes_to_unicode().items()}
    to_bytes = lambda s: bytes(byte_of[c] for c in s)
    id_of = {bytes([i]): i for i in range(256)}
    merges = {}
    for r, (a, b) in enumerate(merge_list):
        ba, bb = to_bytes(a), to_bytes(b)
        new_id = 256 + r
        merges[(id_of[ba], id_of[bb])] = new_id
        id_of[ba + bb] = new_id
    return merges, build_vocab(merges)

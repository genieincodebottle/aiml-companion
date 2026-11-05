"""lltPretrainData shared library: the filters, PII scrub, decontamination and MinHash dedup the lab
page builds step by step. Plain Python and NumPy, so the same functions run in the notebook, in the
Kaggle data-prep kernel and (the NumPy parts) in the browser.

Rule thresholds follow Rae et al. 2021 (Gopher, Appendix A, Table A1) and Raffel et al. 2020 (C4,
section 2.2), with the defaults datatrove 0.10.1 uses for the same rules. Words are split on
whitespace here; datatrove splits with a language tokenizer, so a few borderline documents differ
(the measured agreement is in the Run Library file lltPretrainData-cpu-funnel).
"""
import hashlib
import re
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------------------------
# Gopher quality rules (Rae et al. 2021, Appendix A)
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~")


def gopher_quality(text):
    """Return None if the document passes, else the name of the first rule it fails."""
    words = text.split()
    if not words:
        return "empty"
    content = [w for w in words if any(c not in PUNCT for c in w)]
    if len(content) < 50:
        return "too_few_words"
    if len(content) > 100_000:
        return "too_many_words"
    mean_len = sum(len(w) for w in content) / len(content)
    if mean_len < 3 or mean_len > 10:
        return "mean_word_length"
    if text.count("#") / len(words) > 0.1:
        return "hash_ratio"
    if (text.count("...") + text.count("…")) / len(words) > 0.1:
        return "ellipsis_ratio"
    lines = text.splitlines() or [text]
    if sum(l.lstrip().startswith(("•", "-")) for l in lines) / len(lines) > 0.9:
        return "bullet_lines"
    if sum(l.rstrip().endswith(("...", "…")) for l in lines) / len(lines) > 0.3:
        return "ellipsis_lines"
    if sum(any(c.isalpha() for c in w) for w in words) / len(words) < 0.8:
        return "alpha_words"
    if len(STOP_WORDS & set(w.lower() for w in words)) < 2:
        return "stop_words"
    return None


# ----------------------------------------------------------------------------------------------
# Gopher repetition rules (Rae et al. 2021, Table A1)
TOP_NGRAM = ((2, 0.20), (3, 0.18), (4, 0.16))
DUP_NGRAM = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10))


def _dups(items):
    seen, n_dup, chars = set(), 0, 0
    for x in items:
        if x in seen:
            n_dup += 1
            chars += len(x)
        else:
            seen.add(x)
    return n_dup, chars


def _dup_ngram_chars(words, n):
    # Characters covered by n-grams already seen, skipping ahead past each repeat (as datatrove does).
    seen, chars, i = set(), 0, 0
    while i < len(words) - n + 1:
        g = " ".join(words[i:i + n])
        if g in seen:
            chars += sum(len(w) for w in words[i:i + n])
            i += n
        else:
            seen.add(g)
            i += 1
    return chars


def gopher_repetition(text):
    if not text.strip():
        return "empty"
    paras = re.split("\n{2,}", text.strip())
    n_dup, chars = _dups(paras)
    if n_dup / len(paras) > 0.30:
        return "dup_paragraphs"
    if chars / len(text) > 0.20:
        return "dup_paragraph_chars"
    lines = re.split("\n+", text)
    n_dup, chars = _dups(lines)
    if n_dup / len(lines) > 0.30:
        return "dup_lines"
    if chars / len(text) > 0.20:
        return "dup_line_chars"
    words = text.split()
    for n, frac in TOP_NGRAM:
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        if grams:
            g, c = Counter(grams).most_common(1)[0]
            if len(g) * c / len(text) > frac:
                return "top_%d_gram" % n
    for n, frac in DUP_NGRAM:
        if _dup_ngram_chars(words, n) / len(text) > frac:
            return "dup_%d_gram" % n
    return None


# ----------------------------------------------------------------------------------------------
# C4 rules (Raffel et al. 2020, section 2.2), as the paper states them: keep lines that end in
# terminal punctuation and have at least 5 words, drop pages with fewer than 3 sentences.
# datatrove 0.10.1's C4QualityFilter defaults swap those two numbers (5 sentences, 3 words).
# The bad-words list is not applied here.
POLICY = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")
CITATION = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")
SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def c4_clean(text, terminal_punct=True):
    """Return (cleaned_text, None) if the page is kept, else (None, rule)."""
    kept, sentences = [], 0
    for line in text.splitlines():
        line = line.strip()
        words = line.split()
        if any(len(w) > 1000 for w in words):
            continue
        line = CITATION.sub("", line)
        if terminal_punct and (not line.endswith((".", "?", "!", '"', "'")) or line.endswith("...")):
            continue
        if len(words) < 5:
            continue
        low = line.lower()
        if "lorem ipsum" in low:
            return None, "lorem_ipsum"
        if "javascript" in low:
            continue
        if "{" in line:
            return None, "curly_bracket"
        if any(p in low for p in POLICY):
            continue
        sentences += max(1, len(SENT_END.findall(line)))
        kept.append(line)
    if sentences < 3:
        return None, "fewer_than_3_sentences"
    return "\n".join(kept), None


# ----------------------------------------------------------------------------------------------
# PII scrub: email addresses and public IPv4 addresses, replaced by fixed placeholders.
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\b")


def _private_ip(ip):
    a, b = (int(x) for x in ip.split(".")[:2])
    return a in (0, 10, 127) or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31) or (a == 169 and b == 254) or a >= 224


def scrub_pii(text):
    """Return (text, n_emails, n_ips) with emails and public IPv4s replaced."""
    text, n_email = EMAIL.subn("<EMAIL>", text)
    n_ip = 0

    def ip_sub(m):
        nonlocal n_ip
        if _private_ip(m.group(0)):
            return m.group(0)
        n_ip += 1
        return "<IP>"

    text = IPV4.sub(ip_sub, text)
    return text, n_email, n_ip


# ----------------------------------------------------------------------------------------------
# n-gram decontamination: drop a document that shares any n-gram (13 words, lower-cased) with an
# evaluation set.
WORD = re.compile(r"[a-z0-9]+")


def ngrams(text, n=13):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_eval_index(eval_texts, n=13):
    idx = set()
    for t in eval_texts:
        idx |= ngrams(t, n)
    return idx


def contaminated(text, index, n=13):
    return not ngrams(text, n).isdisjoint(index)


# ----------------------------------------------------------------------------------------------
# Exact dedup on normalised text
def norm_text(text):
    return " ".join(WORD.findall(text.lower()))


def exact_key(text):
    return hashlib.sha1(norm_text(text).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------
# MinHash + LSH in NumPy. Shingles are word 5-grams of the normalised text, hashed to 31 bits.
# Each of the K hash functions is h(x) = (a * x + b) mod P with P = 2**31 - 1, so a * x stays
# below 2**62 and fits in uint64.
P31 = np.uint64(2**31 - 1)


def shingles(text, n=5):
    w = norm_text(text).split()
    grams = {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF
                     for g in grams], dtype=np.uint64)


def make_hashes(k, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31 - 1, size=k, dtype=np.uint64)
    b = rng.integers(0, 2**31 - 1, size=k, dtype=np.uint64)
    return a, b


def minhash(sh, a, b):
    if sh.size == 0:
        return np.full(a.size, 2**31 - 1, dtype=np.uint64)
    return ((sh[:, None] * a[None, :] + b[None, :]) % P31).min(axis=0)


def lsh_candidates(sigs, bands, rows):
    """sigs: (n_docs, bands*rows). Return the set of (i, j) pairs that share at least one band."""
    pairs = set()
    for bnd in range(bands):
        buckets = {}
        for i, s in enumerate(sigs[:, bnd * rows:(bnd + 1) * rows]):
            buckets.setdefault(s.tobytes(), []).append(i)
        for ids in buckets.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add((ids[x], ids[y]))
    return pairs


def p_candidate(j, bands, rows):
    """Probability that two documents with Jaccard similarity j share at least one band."""
    return 1 - (1 - j ** rows) ** bands


def clusters(n, pairs):
    """Union-find over candidate pairs. Returns the root of each document; keep one per root."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def jaccard(s1, s2):
    s1, s2 = set(s1.tolist()), set(s2.tolist())
    return len(s1 & s2) / max(1, len(s1 | s2))

"""Extend Qwen3's tokenizer for Hindi by continuing BPE on its own segmentation.

extend(base_id, texts, n_new, mark_regex, out_dir, pin_special=True) -> (out_dir, report).
Used by run_extend.py, measure.py and the notebook (the notebook cell is the same code).

mark_regex=True also changes the pre-tokenizer regex so a Devanagari vowel sign (Unicode category M)
stays inside its word, as Qwen3.5's tokenizer does. Every original merge keeps its rank, and every new
merge goes after them and is learned from Devanagari pieces only, so text without Devanagari tokenizes exactly as before.

pin_special=False reproduces the broken run: the `tokenizers` library numbers added tokens from the size
of the model vocabulary, so 10,000 new entries push <|endoftext|> from 151643 to 161643, onto a new token.
The fix writes each special token into the model vocabulary with its old id.
No backslash appears in this file's regex edits: B = chr(92) builds them, the same trick the page code uses.
"""
import json, os, collections, hashlib
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import AutoTokenizer

B = chr(92)                      # a backslash
L, M, N = B + "p{L}", B + "p{M}", B + "p{N}"


def mark_regex_edit(regex):
    # Qwen3: a run of letters may not contain a vowel sign. Qwen3.5: letters and marks together.
    old1, new1 = "?" + L + "+", "?[" + L + M + "]+"
    old2, new2 = "[^" + B + "s" + L + N + "]+", "[^" + B + "s" + L + M + N + "]+"
    assert old1 in regex and old2 in regex, "unexpected pre-tokenizer regex"
    return regex.replace(old1, new1).replace(old2, new2)


def extend(base_id, texts, n_new, mark_regex, out_dir, pin_special=True):
    tok = AutoTokenizer.from_pretrained(base_id)
    j = json.loads(tok.backend_tokenizer.to_str())
    if mark_regex:
        split = j["pre_tokenizer"]["pretokenizers"][0]["pattern"]
        split["Regex"] = mark_regex_edit(split["Regex"])
    work = Tokenizer.from_str(json.dumps(j))          # original vocab and merges, maybe the new regex
    vocab = j["model"]["vocab"]                       # byte-level string -> id
    inv = {i: s for s, i in vocab.items()}
    # 1. Split the corpus into pieces with the pre-tokenizer and count each Hindi piece.
    # Only pieces made entirely of Devanagari (U+0900 to U+097F, after one optional leading space) count,
    # so English, digits, punctuation and code keep their tokens.
    pieces = collections.Counter()
    for t in texts:
        t = work.normalizer.normalize_str(t)
        for p, (a, b) in work.pre_tokenizer.pre_tokenize_str(t):
            w = t[a:b].lstrip(" ")
            if w and all(0x900 <= ord(ch) <= 0x97F for ch in w):
                pieces[p] += 1
    # 2. Segment each piece with the ORIGINAL merges, write it as one private-use character per token, run BPE on that.
    char_of, seg = {}, {}
    for p in pieces:
        seg[p] = [x.id for x in work.model.tokenize(p)]
        for i in seg[p]:
            if i not in char_of:
                char_of[i] = chr(0xF0000 + len(char_of))
    id_of_char = {c: i for i, c in char_of.items()}
    multi = [(p, c) for p, c in pieces.items() if len(seg[p]) > 1]

    def corpus():
        for p, c in multi:
            w = "".join(char_of[i] for i in seg[p])
            for _ in range(c):
                yield w
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    alphabet = list(char_of.values())
    trainer = trainers.BpeTrainer(vocab_size=len(alphabet) + n_new, min_frequency=2, initial_alphabet=alphabet,
                                  show_progress=False, limit_alphabet=len(alphabet))
    if n_new > 0:
        bpe.train_from_iterator(corpus(), trainer=trainer, length=sum(c for _, c in multi))
        merges = json.loads(bpe.to_str())["model"]["merges"]
    else:
        merges = []
    to_bytes = lambda s: "".join(inv[id_of_char[ch]] for ch in s)
    # 3. Append each learned merge after the original ones, with a new id for every new string.
    first_new = max(a["id"] for a in j["added_tokens"]) + 1     # 151669 for Qwen3
    if pin_special:
        for a in j["added_tokens"]:
            vocab[a["content"]] = a["id"]                        # keeps <|endoftext|> at 151643
    next_id, new_tokens = first_new, {}
    for m in merges:
        a, b = (m.split(" ") if isinstance(m, str) else m)
        sa, sb = to_bytes(a), to_bytes(b)
        if sa + sb not in vocab:
            vocab[sa + sb] = next_id
            new_tokens[sa + sb] = next_id
            next_id += 1
        j["model"]["merges"].append([sa, sb])
    os.makedirs(out_dir, exist_ok=True)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False)
    # 4. For mean-of-subword init: the original token ids each new token is made of.
    orig = Tokenizer.from_str(tok.backend_tokenizer.to_str()).model
    parts = {i: [x.id for x in orig.tokenize(s)] for s, i in new_tokens.items()}
    with open(os.path.join(out_dir, "new_token_parts.json"), "w") as f:
        json.dump({str(k): v for k, v in parts.items()}, f)
    h = hashlib.sha256(json.dumps(j["model"]["merges"][len(j["model"]["merges"]) - len(merges):]).encode()).hexdigest()[:16]
    return out_dir, {"pieces": len(pieces), "alphabet": len(alphabet), "mergesAdded": len(merges),
                     "newTokens": len(new_tokens), "firstNewId": first_new, "lastNewId": next_id - 1, "mergesSha": h,
                     "pinSpecial": pin_special, "markRegex": mark_regex}

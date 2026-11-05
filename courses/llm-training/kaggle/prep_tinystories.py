"""Data prep for the lltTinyStories sweep: train the BPE tokenizer on the TRAIN split only, encode
both splits once, and cache the token ids as uint16 .npy files that every training run reuses.

Usage: python prep_tinystories.py <cache_dir> <work_dir> [max_train_stories]
"""
import json, os, sys, time
import numpy as np

cache_dir, work_dir = sys.argv[1], sys.argv[2]
max_train = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(cache_dir, exist_ok=True); os.makedirs(work_dir, exist_ok=True)
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

T0 = time.time()
ds = load_dataset("roneneldan/TinyStories")
train_text, val_text = ds["train"]["text"], ds["validation"]["text"]
if max_train: train_text = train_text[:max_train]
load_sec = time.time() - T0

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
t = time.time(); tok.train_from_iterator(train_text[:300000], trainer=trainer); tok_sec = time.time() - t
EOT = tok.token_to_id("<|eot|>")
tok.save(os.path.join(cache_dir, "tokenizer.json")); tok.save(os.path.join(work_dir, "tokenizer.json"))

enc = getattr(tok, "encode_batch_fast", None) or tok.encode_batch


def encode_all(texts):
    out = []
    for e in enc(texts):
        out.append(np.asarray(e.ids, dtype=np.uint16)); out.append(np.array([EOT], dtype=np.uint16))
    return np.concatenate(out)


t = time.time()
train_ids = np.concatenate([encode_all(train_text[i:i + 100000]) for i in range(0, len(train_text), 100000)])
val_ids = encode_all(val_text)
encode_sec = time.time() - t

half = len(val_text) // 2
val_b_start = int(encode_all(val_text[:half]).size)

# Exact byte length of every token id. Byte-level BPE maps each byte to one printable character
# (the GPT-2 table), so a token's characters map back to its bytes one for one.
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
V = tok.get_vocab_size()
token_bytes = np.zeros(V, dtype=np.int64)
for i in range(V):
    s = tok.id_to_token(i)
    token_bytes[i] = 1 if i == EOT else len(s)
    assert i == EOT or all(ch in char_to_byte for ch in s), s
# Check: token bytes over the val split equal the UTF-8 bytes of the text, plus one per story.
val_bytes = sum(len(s.encode()) for s in val_text)
assert int(token_bytes[val_ids].sum()) == val_bytes + len(val_text), (int(token_bytes[val_ids].sum()), val_bytes)

t = time.time()
np.save(os.path.join(cache_dir, "train_ids.npy"), train_ids); np.save(os.path.join(cache_dir, "val_ids.npy"), val_ids)
np.save(os.path.join(cache_dir, "token_bytes.npy"), token_bytes)
save_sec = time.time() - t
t = time.time(); _ = np.load(os.path.join(cache_dir, "train_ids.npy")); _ = np.load(os.path.join(cache_dir, "val_ids.npy")); load_cache_sec = time.time() - t

meta = {"eot": EOT, "valBStartToken": val_b_start,
        "data": {"dataset": "roneneldan/TinyStories", "trainStories": len(train_text), "valStories": len(val_text),
                 "trainTokens": int(train_ids.size), "valTokens": int(val_ids.size),
                 "valAStories": half, "valBStartToken": val_b_start,
                 "bytesPerToken": round(val_bytes / (val_ids.size - len(val_text)), 3)},
        "timing": {"datasetLoadSec": round(load_sec, 1), "tokenizerTrainSec": round(tok_sec, 1),
                   "encodeSec": round(encode_sec, 1), "encoder": enc.__name__,
                   "cacheSaveSec": round(save_sec, 1), "cacheLoadSec": round(load_cache_sec, 1),
                   "cpuCount": os.cpu_count()}}
json.dump(meta, open(os.path.join(cache_dir, "meta.json"), "w"), indent=1)
json.dump(meta, open(os.path.join(work_dir, "prep_meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))

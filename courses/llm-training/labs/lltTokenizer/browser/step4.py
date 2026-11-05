import random
from collections import Counter

text = "The tokenizer reads text as bytes. टोकनाइज़र टेक्स्ट को बाइट्स के रूप में पढ़ता है। El tokenizador lee el texto como bytes."

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

ids, merges = list(text.encode("utf-8")), {}
for new_id in range(256, 256 + 40):  # step 2's trainer
    pairs = Counter(zip(ids, ids[1:]))
    best = max(pairs, key=pairs.get)
    merges[best] = new_id
    ids = merge(ids, best, new_id)
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():
    vocab[new_id] = vocab[a] + vocab[b]

def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) > 1:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8")

print(encode("the text"), decode(encode("the text")))
rng = random.Random(0)
failures = 0
for _ in range(1000):
    chars = [rng.choice([rng.randrange(32, 127), rng.randrange(0x900, 0x980), rng.randrange(0xE000, 0x110000)])
             for _ in range(rng.randrange(0, 30))]
    s = "".join(map(chr, chars))
    failures += decode(encode(s)) != s
print("round-trip failures in 1000 random strings:", failures)

import time
from collections import Counter

text = """The tokenizer reads text as bytes. It counts every pair of neighbouring bytes and merges the most frequent pair into a new token. Then it counts again. After a few hundred merges, common words such as the, and, token and text are single tokens.
टोकनाइज़र टेक्स्ट को बाइट्स के रूप में पढ़ता है। वह पास-पास के हर जोड़े को गिनता है और सबसे ज़्यादा आने वाले जोड़े को एक नए टोकन में जोड़ देता है। फिर वह दोबारा गिनता है। कुछ सौ मर्ज के बाद आम शब्द एक-एक टोकन बन जाते हैं।
El tokenizador lee el texto como bytes. Cuenta cada par de bytes vecinos y une el par más frecuente en un token nuevo. Después vuelve a contar. Tras unos cientos de fusiones, las palabras comunes como el, de, token y texto son un solo token."""

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

def train(ids, num_merges):
    merges = {}
    for new_id in range(256, 256 + num_merges):
        pairs = Counter(zip(ids, ids[1:]))
        best = max(pairs, key=pairs.get)
        merges[best] = new_id
        ids = merge(ids, best, new_id)
    return merges, ids

start = time.perf_counter()
data = list(text.encode("utf-8"))
merges, ids = train(data, 100)
print(f"{len(data)} bytes -> {len(ids)} tokens in {time.perf_counter() - start:.2f} s")

vocab = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():
    vocab[new_id] = vocab[a] + vocab[b]
print([vocab[i].decode("utf-8", "replace") for i in range(256, 266)])
crossing = [vocab[i] for i in merges.values() if b" " in vocab[i].lstrip(b" ")]
print(len(crossing), "of 100 new tokens hold a space after a non-space byte")
print([t.decode("utf-8", "replace") for t in crossing if t.isascii()][:6])

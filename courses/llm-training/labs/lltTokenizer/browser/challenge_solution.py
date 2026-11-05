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

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) > 1:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

# Checks. Run the cell; it prints All checks passed when both functions are right.
ids = [1, 2, 3, 1, 2]
assert merge(ids, (1, 2), 99) == [99, 3, 99]
assert ids == [1, 2, 3, 1, 2], "merge changed its input list"
assert merge([5, 5, 5], (5, 5), 7) == [7, 5], "read pairs left to right, without overlap"
assert merge([5, 5, 5, 5], (5, 5), 7) == [7, 7]
assert merge([], (1, 2), 3) == [] and merge([1], (1, 2), 3) == [1]
assert merge([1, 2], (2, 1), 9) == [1, 2]

# Ranks: ll is merge 256, he is 257, hel is 258, hell is 259 (dict order is not rank order).
merges = {(257, 256): 259, (104, 101): 257, (108, 108): 256, (257, 108): 258}
assert encode("", merges) == []
assert encode("hi", merges) == [104, 105]
got = encode("hello", merges)
assert got != [258, 108, 111], "merged the leftmost pair first; merge the lowest rank first"
assert got != [257, 256, 111], "applied merges in dict order; use the rank (the new id)"
assert got == [259, 111], got
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), i in sorted(merges.items(), key=lambda kv: kv[1]):
    vocab[i] = vocab[a] + vocab[b]
assert b"".join(vocab[i] for i in encode("hello hell", merges)).decode("utf-8") == "hello hell"
assert encode("नमस्ते", merges) == list("नमस्ते".encode("utf-8"))
print("All checks passed")

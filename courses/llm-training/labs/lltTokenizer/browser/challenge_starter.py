def merge(ids, pair, new_id):
    # Return a NEW list where every non-overlapping (a, b) == pair, read left to right,
    # is replaced by new_id. Do not change ids.
    raise NotImplementedError

def encode(text, merges):
    # merges maps (a, b) -> new_id, and a smaller new_id means an earlier merge.
    # Start from the UTF-8 bytes of text. While some neighbouring pair is in merges,
    # merge the pair with the smallest new_id. Return the list of ids.
    raise NotImplementedError

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

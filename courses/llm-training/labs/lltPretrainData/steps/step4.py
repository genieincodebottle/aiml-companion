import collections
import ptd  # the notebook writes ptd.py in its first cells

removed = collections.Counter()
kept = []
for url, text in english:
    rule = ptd.gopher_quality(text) or ptd.gopher_repetition(text)
    cleaned = None
    if rule is None:
        cleaned, rule = ptd.c4_clean(text)
    if rule:
        removed[rule] += 1
    else:
        kept.append((url, cleaned))
for rule, n in removed.most_common():
    print(f"{rule:24} {n:5}")
print("kept", len(kept), "of", len(english))
words_in = sum(len(t.split()) for _, t in english)
words_out = sum(len(t.split()) for _, t in kept)
print("words kept", words_out, "of", words_in)

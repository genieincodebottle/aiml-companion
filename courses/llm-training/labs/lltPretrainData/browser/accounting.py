import numpy as np

# Measured on our corpora (run lltPretrainData-cpu-corpora): mean tokens per document,
# and the unique tokens we tokenized, for web (filtered C4), Wikipedia (wikitext-103) and stories.
names = ["web", "wikipedia", "stories"]
mean_doc_tokens = np.array([922.6, 5578.2, 248.2])
unique_tokens = np.array([55_141_258, 60_004_232, 39_712_508])
budget = 300 * 64 * 512  # the mixture runs, 300 steps of 64 x 512 tokens

def token_share_from_doc_weights(doc_w, mean_len):
    t = doc_w * mean_len
    return t / t.sum()

def effective_epochs(token_w, budget, unique):
    return token_w * budget / unique

equal = np.array([1 / 3, 1 / 3, 1 / 3])
by_docs = token_share_from_doc_weights(equal, mean_doc_tokens)
for n, s in zip(names, by_docs):
    print(f"equal weights by document -> {n:9} gets {s:.1%} of the tokens")
for n, e in zip(names, effective_epochs(equal, budget, unique_tokens)):
    print(f"equal weights by token    -> {n:9} is seen {e:.3f} times")

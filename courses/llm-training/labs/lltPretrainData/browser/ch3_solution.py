import numpy as np

def effective_epochs(token_weights, budget, unique_tokens):
    # How many times the run reads each source: its share of the budget over its unique tokens
    return np.asarray(token_weights) * budget / np.asarray(unique_tokens)

def token_weights_from_docs(doc_weights, mean_doc_tokens):
    # Turn weights over documents into weights over tokens
    t = np.asarray(doc_weights) * np.asarray(mean_doc_tokens)
    return t / t.sum()

def over_limit(token_weights, budget, unique_tokens, limit=4.0):
    # Indices of the sources read more than limit times
    e = effective_epochs(token_weights, budget, unique_tokens)
    return [i for i, x in enumerate(e) if x > limit]

# Checks. Run the cell. It prints All checks passed when all three functions are right.
budget = 2_000_000_000
unique = [6_000_000_000, 300_000_000, 50_000_000]   # web, Wikipedia, textbooks
w = [0.70, 0.20, 0.10]
e = effective_epochs(w, budget, unique)
assert e is not None, "effective_epochs returns None"
assert np.allclose(e, [0.2333, 1.3333, 4.0], atol=1e-3), f"epochs should be 0.233, 1.333, 4.0, got {e}"
tw = token_weights_from_docs([0.5, 0.25, 0.25], [500, 3000, 200])
assert tw is not None, "token_weights_from_docs returns None"
assert np.allclose(tw, [0.2381, 0.7143, 0.0476], atol=1e-3), f"token weights should be 0.238, 0.714, 0.048, got {tw}"
assert over_limit(w, budget, unique) == [], "4.0 epochs is at the limit, not over it"
assert over_limit([0.6, 0.2, 0.2], budget, unique) == [2], "textbooks at 8 epochs are over the limit"
print("epochs", np.round(e, 3), "| token weights", np.round(tw, 3))
print("All checks passed")

import numpy as np

# A toy chat tokenizer. Special ids first, then one id per word.
IM_START, IM_END, NL = 1, 2, 3
ROLE = {"system": 4, "user": 5, "assistant": 6}
WORDS = {}

def word_id(w):
    return WORDS.setdefault(w, 10 + len(WORDS))

def render(chat):
    # Each turn: <|im_start|> role NL words... <|im_end|> NL, as in the SmolLM2 template
    ids = []
    for role, text in chat:
        ids += [IM_START, ROLE[role], NL] + [word_id(w) for w in text.split()] + [IM_END, NL]
    return np.array(ids)

def build_labels(ids):
    # Return an array like ids: the token id inside each assistant reply and its <|im_end|>, -100 everywhere else.
    # The header IM_START, ROLE["assistant"], NL and the NL after <|im_end|> get -100.
    labels = np.full(len(ids), -100)
    t = 0
    while t < len(ids):
        if t + 2 < len(ids) and ids[t] == IM_START and ids[t + 1] == ROLE["assistant"] and ids[t + 2] == NL:
            t += 3
            while t < len(ids):
                labels[t] = ids[t]
                t += 1
                if ids[t - 1] == IM_END:
                    break
        else:
            t += 1
    return labels

def masked_mean_nll(logp, labels):
    # logp: (T, V) log-probabilities; row t predicts token t + 1. Mean NLL over the labels that are not -100.
    target = labels[1:]
    keep = target != -100
    return float(-logp[:-1][keep, target[keep]].mean())

def doc_mask(lengths):
    # (T, T) boolean: True where query i may attend to key j. Causal, and only inside its own document.
    doc = np.repeat(np.arange(len(lengths)), lengths)
    T = len(doc)
    return np.tril(np.ones((T, T), dtype=bool)) & (doc[:, None] == doc[None, :])

# Checks. Run the cell; it prints All checks passed when all three functions are right.
chat = [("system", "be brief"), ("user", "name a fruit"), ("assistant", "a mango"),
        ("user", "and a colour"), ("assistant", "red")]
ids = render(chat)
labels = build_labels(ids)
assert labels.shape == ids.shape, "return one label per token"
assert labels.tolist() == [-100] * 18 + [WORDS["a"], WORDS["mango"], IM_END] + [-100] * 12 + [WORDS["red"], IM_END, -100], labels.tolist()
assert (labels != -100).sum() == 5, "two replies of 2 and 1 words, each with its <|im_end|>"
assert build_labels(render([("user", "hi"), ("user", "again")])).max() == -100, "no assistant turn means no loss"
V = 40
logits = np.sin(np.arange(len(ids) * V).reshape(len(ids), V) * 0.37)
logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
loss = masked_mean_nll(logp, labels)
assert abs(loss - 4.12037) < 1e-4, round(loss, 5)
assert abs(masked_mean_nll(logp, ids) - 3.86735) < 1e-4, "with labels = ids every next token counts: " + str(round(masked_mean_nll(logp, ids), 5))
m = doc_mask([3, 2])
assert m.dtype == bool and m.shape == (5, 5), "a (T, T) boolean array"
assert m.astype(int).tolist() == [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]], m.astype(int).tolist()
assert doc_mask([4]).sum() == 10, "one document is the plain causal mask"
print("loss tokens", int((labels != -100).sum()), "of", len(ids), "| masked NLL", round(loss, 4))
print("All checks passed")

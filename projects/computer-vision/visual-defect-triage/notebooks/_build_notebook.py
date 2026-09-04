"""Builds the standalone notebook as plain text, so it diffs like code.

The notebook is SELF-CONTAINED on purpose. Kaggle cannot import src/, so the
logic is inlined rather than imported. That means this file and src/ can drift,
which is why the last cell re-derives the same three headline numbers the repo's
own pipeline reports, and asserts them.

    python notebooks/_build_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent / "visual_defect_triage_standalone.ipynb"

nb = nbf.v4.new_notebook()
C: list = []


def md(text: str) -> None:
    C.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    C.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- 0. intro
md("""
# Visual Defect Triage

A camera photographs every part coming off a production line. Something has to
decide whether each part passes, fails, or needs a human to look at it.

That looks like a classification task until you try to ship it. A classifier
gives you a class and a number, and shipping means deciding what to do when the
number is 0.71. This notebook builds the parts around the model that turn a
prediction into a decision.

**What you will build**

1. A dataset where one defect is rare and genuinely hard, because that is what
   makes evaluation interesting
2. A linear head on frozen embeddings, which is the baseline fine-tuning has to beat
3. Temperature scaling, so the confidence is worth putting a threshold on
4. Per-slice evaluation ranked by improvement ceiling, not by error rate
5. A routing gate with a policy override no confidence can bypass
6. Nearest-neighbour retrieval from the same embedding, for the human reviewer

Everything runs on CPU in under a minute. No downloads.

> Full walkthrough and narrated overview at
> [AI-ML Companion](https://aimlcompanion.ai/module/computerVision/cvVitCapstone).
""")

code("""
import numpy as np
from collections import Counter, defaultdict

SEED = 13
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True)
print("ready")
""")

# ---------------------------------------------------------------- 1. data
md("""
## 1. A dataset with a rare, subtle defect

Real defect data is not seven equally difficult classes. Most defects are
visually obvious and one or two are subtle, and the subtle ones are usually rare.

`hairline_crack` is modelled as a small offset from the `pass` centre rather than
its own well-separated cluster. That is physically true, a hairline crack looks
like a good part until you look properly, and it is what makes the evaluation
lesson later visible.

Only 3,000 of the training images carry a label. Labels are the expensive thing
in this problem, and that budget is also what makes the model overconfident.
""")

code("""
CLASSES = ["pass", "scratch", "dent", "discolour", "contamination",
           "weld_void", "hairline_crack"]
SHARES = np.array([0.62, 0.14, 0.09, 0.06, 0.04, 0.02, 0.03])
SHARES = SHARES / SHARES.sum()

DIM, N = 96, 12000
EASY_SEP, CRACK_DELTA, NOISE = 7.0, 3.6, 1.15
LABEL_BUDGET = 3000

centres = rng.normal(0, 1, size=(len(CLASSES), DIM))
centres = centres / np.linalg.norm(centres, axis=1, keepdims=True) * EASY_SEP

# The subtle one sits next to pass.
off = rng.normal(0, 1, DIM)
centres[CLASSES.index("hairline_crack")] = (
    centres[CLASSES.index("pass")] + off / np.linalg.norm(off) * CRACK_DELTA)

labels = rng.choice(len(CLASSES), size=N, p=SHARES)
embeddings = centres[labels] + rng.normal(0, NOISE, size=(N, DIM))

# Photographs come in production batches of 25. This matters in the next cell.
batch_id = np.arange(N) // 25

print({c: int((labels == i).sum()) for i, c in enumerate(CLASSES)})
""")

# ---------------------------------------------------------------- 2. splits
md("""
## 2. Split by batch, never by image

A line photographs the same part several times, so those images are
near-duplicates. Split by image and one copy lands in train while another lands
in test, and the test score then reports memory rather than generalisation.

Group by whatever unit produced the duplicates, here the production batch, and
split those groups.
""")

code("""
def split_by_batch(batch_id, seed=13, val=0.15, test=0.15):
    batches = np.unique(batch_id)
    np.random.default_rng(seed).shuffle(batches)
    n_val, n_test = int(len(batches) * val), int(len(batches) * test)
    sets = {"test": batches[:n_test],
            "val": batches[n_test:n_test + n_val],
            "train": batches[n_test + n_val:]}
    return {k: np.where(np.isin(batch_id, v))[0] for k, v in sets.items()}

idx = split_by_batch(batch_id)

# The property the split exists to guarantee, asserted rather than assumed.
for a in ("train", "val", "test"):
    for b in ("train", "val", "test"):
        if a < b:
            overlap = set(batch_id[idx[a]]) & set(batch_id[idx[b]])
            assert not overlap, f"batch leak between {a} and {b}"

print({k: len(v) for k, v in idx.items()}, "no batch appears in two splits")
""")

# ---------------------------------------------------------------- 3. head
md("""
## 3. A linear head on frozen features

Start with the cheapest thing that could work. A single linear layer on frozen
embeddings trains in seconds, is hard to overfit, and tells you how much of the
task the pretrained features already solve. It is the baseline that says whether
full fine-tuning bought anything.

Class weights are raised to the power 0.5 rather than 1.0. Full inverse-frequency
weighting buys `hairline_crack` recall by pushing false cracks onto `pass`, and
`pass` is 62 percent of traffic, so that trade is worth measuring.
""")

code("""
def softmax(z, t=1.0):
    z = np.asarray(z, dtype="float64") / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def train(x, y, k, epochs=1500, lr=0.5, wd=1e-6, weight_power=0.5, seed=0):
    n, d = x.shape
    r = np.random.default_rng(seed)
    w, b = r.normal(0, 0.01, size=(k, d)), np.zeros(k)

    counts = np.bincount(y, minlength=k).astype("float64")
    counts[counts == 0] = 1.0
    sw = ((counts.sum() / (k * counts)) ** weight_power)[y]
    sw = sw / sw.sum()
    onehot = np.eye(k)[y]

    for _ in range(epochs):
        err = (softmax(x @ w.T + b) - onehot) * sw[:, None]
        w -= lr * (err.T @ x + wd * w)
        b -= lr * err.sum(axis=0)
    return w, b

labelled = idx["train"][:LABEL_BUDGET]
w, b = train(embeddings[labelled], labels[labelled], len(CLASSES))
print(f"fitted on {len(labelled)} labelled of {len(idx['train'])} train images")
""")

# ---------------------------------------------------------------- 4. calibration
md("""
## 4. Temperature scaling

The gate is a threshold on confidence, so the confidence has to mean something.
Straight out of training it does not: the model claims 0.95 on images it gets
right far less often than that.

Temperature scaling divides every logit by one number fitted on **validation**.
Because every logit is divided by the same positive value the ranking cannot
change, so the predicted class is identical for every image and accuracy is
untouched. Only the claimed certainty moves.

Fit it on validation and never on test. Fitting on test means the number you
quote was chosen using the data you quote it on.
""")

code("""
def nll(logits, y, t):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))).mean())

def fit_temperature(logits, y, lo=0.05, hi=10.0, iters=60):
    phi = (np.sqrt(5) - 1) / 2
    a, bb = lo, hi
    c, d = bb - phi * (bb - a), a + phi * (bb - a)
    for _ in range(iters):
        if nll(logits, y, c) < nll(logits, y, d):
            bb, d = d, c; c = bb - phi * (bb - a)
        else:
            a, c = c, d; d = a + phi * (bb - a)
    return (a + bb) / 2

def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)

T = fit_temperature(embeddings[idx["val"]] @ w.T + b, labels[idx["val"]])

test_logits = embeddings[idx["test"]] @ w.T + b
y_test = labels[idx["test"]]
raw, cal = softmax(test_logits, 1.0), softmax(test_logits, T)
pred = cal.argmax(1)
correct = pred == y_test
conf = cal[np.arange(len(pred)), pred]

assert (raw.argmax(1) == pred).all(), "temperature changed a prediction"

print(f"temperature {T:.3f}   (above 1 means the model was overconfident)")
print(f"accuracy    {correct.mean():.3f}   (identical before and after, by construction)")
print(f"ECE {ece(raw.max(1), correct):.4f} -> {ece(conf, correct):.4f}")
""")

# ---------------------------------------------------------------- 5. slices
md("""
## 5. One number cannot run a factory

Overall accuracy hides the slice that matters. Break it out by class, and rank by
**improvement ceiling**, which is share multiplied by error rate. That is how
much overall accuracy would gain if the slice became perfect.

Watch which row comes top. It is not the worst class.
""")

code("""
rows = []
for i, name in enumerate(CLASSES):
    m = y_test == i
    if not m.any():
        continue
    n, acc = int(m.sum()), float(correct[m].mean())
    share = n / len(y_test)
    rows.append({"slice": name, "n": n, "share": share,
                 "accuracy": acc, "ceiling": share * (1 - acc)})
rows.sort(key=lambda r: -r["ceiling"])

overall = float(correct.mean())
total_ceiling = sum(r["ceiling"] for r in rows)
assert abs(total_ceiling - (1 - overall)) < 1e-9, "ceilings must sum to the error budget"

print(f"{'slice':<16}{'n':>6}{'share':>9}{'accuracy':>10}{'ceiling':>10}")
for r in rows:
    print(f"{r['slice']:<16}{r['n']:>6}{r['share']:>8.1%}{r['accuracy']:>10.3f}{r['ceiling']:>10.4f}")
print()
print(f"ceilings sum to {total_ceiling:.4f} = 1 - {overall:.4f}, the whole error budget")
""")

md("""
`hairline_crack` is far and away the worst class. It also offers a **smaller
ceiling** than `pass`, which everyone considers healthy, because `pass` carries
roughly 64 percent of traffic and a ceiling is error *mass*, not error *rate*.

Sort by rate and you work on the wrong slice. That does not mean ignore the crack,
whose accuracy may be unacceptable on its own terms. It means the argument gets
made on ceiling and cost rather than on rate alone.
""")

# ---------------------------------------------------------------- 6. gate
md("""
## 6. The gate, and the rule no confidence can bypass

Routing is a business rule rather than a model output. Above the accept threshold
the part goes through, below the reject threshold it is scrapped, and everything
between goes to a human.

Two classes ignore the confidence entirely. A structural failure reaching a
customer is not comparable to a scratch reaching one, so `hairline_crack` and
`weld_void` always reach a reviewer whatever the model says.

That is the rule an optimisation removes. Someone measuring review volume will
notice these are nearly always right at 0.999 and widening the gate looks like
free money.
""")

code("""
ACCEPT_ABOVE, REJECT_BELOW = 0.98, 0.02
NEVER_AUTO = {"hairline_crack", "weld_void"}

def route(cls_name, confidence):
    if cls_name in NEVER_AUTO:
        return "review"
    if confidence >= ACCEPT_ABOVE:
        return "auto_accept"
    if confidence <= REJECT_BELOW:
        return "auto_reject"
    return "review"

routes = [route(CLASSES[p], float(c)) for p, c in zip(pred, conf)]
counts = Counter(routes)

accepted = conf >= ACCEPT_ABOVE
escaped = int((accepted & ~correct).sum())

for k in ("auto_accept", "auto_reject", "review"):
    print(f"{k:<14}{counts[k] / len(routes):>7.1%}")
print()
print(f"escaped errors {escaped} of {int(accepted.sum())} accepted "
      f"({escaped / max(int(accepted.sum()), 1):.2%})")
print()
print(f"{'accept_above':>14}{'accept share':>15}{'escaped':>10}")
for t in (0.90, 0.95, 0.98, 0.99):
    a = conf >= t
    print(f"{t:>14.2f}{a.mean():>15.1%}{int((a & ~correct).sum()):>10}")
""")

md("""
That sweep is how the threshold gets chosen. It is not a modelling decision, it
is a trade between reviewer cost and escaped defects, and turning it into a table
lets someone with the authority pick a row.
""")

# ---------------------------------------------------------------- 7. retrieval
md("""
## 7. The embedding's second job

The vectors already exist, so indexing them costs nothing extra. Normalise them
and an inner product is cosine similarity.

This changes the reviewer's question. A classifier tells them the model thinks
this is a scratch with confidence 0.71, which they already doubted or it would
not be in their queue. Retrieval shows them the most similar past parts and how
each was ruled on, which turns an open judgement into a comparison against
precedent.
""")

code("""
def normalise(x):
    x = np.atleast_2d(np.asarray(x, dtype="float64"))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

index = normalise(embeddings)

def neighbours(vec, k=8):
    sims = index @ normalise(vec)[0]
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), CLASSES[labels[i]]) for i in top]

query = idx["test"][int(np.argmin(conf))]        # the least confident test image
print(f"query image {query}, true class {CLASSES[labels[query]]}, "
      f"model said {CLASSES[pred[int(np.argmin(conf))]]} at {conf.min():.3f}")
print()
for i, sim, ruling in neighbours(embeddings[query]):
    print(f"  img {i:<7} similarity {sim:.3f}   ruled: {ruling}")
""")

# ---------------------------------------------------------------- 8. loop
md("""
## 8. The loop, and the trap inside it

Every reviewer decision is a labelled example, drawn from exactly the part of the
distribution the model finds hard. That is the flywheel, and it is worth more
than the compute by a wide margin.

There is one way to break it and it is silent. Mined images are harder than
production traffic by construction, because they were selected for being
uncertain. Let them into the evaluation set and the reported score drifts away
from reality, and it drifts further every cycle.
""")

code("""
review_idx = idx["test"][np.array([r == "review" for r in routes])]
mined = set(int(i) for i in review_idx)
frozen_eval = set(int(i) for i in idx["test"]) - mined

def assert_not_in_evaluation(mined_ids, eval_ids):
    leaked = mined_ids & eval_ids
    if leaked:
        raise AssertionError(f"mined images leaked into evaluation: {sorted(leaked)[:5]}")

assert_not_in_evaluation(mined, frozen_eval)
print(f"{len(mined)} mined for training, {len(frozen_eval)} held frozen for measurement")

# Drift needs no labels, which is why it is the signal that moves first.
def population_shift(ref, win):
    a, b_ = np.asarray(ref).mean(0), np.asarray(win).mean(0)
    return float(1 - (a @ b_) / (np.linalg.norm(a) * np.linalg.norm(b_)))

ref = embeddings[idx["train"]]
print(f"drift, same distribution : {population_shift(ref, embeddings[idx['test']]):.4f}")
print(f"drift, shifted lighting  : {population_shift(ref, embeddings[idx['test']] + 0.9):.4f}")
""")

# ---------------------------------------------------------------- 9. recap
md("""
## What to take away

- **Split by the unit that produced the duplicates**, not by row. Near-duplicates
  across a split boundary turn a test score into a memory test.
- **Calibrate before you threshold.** Temperature scaling costs one parameter,
  cannot change a prediction, and is what makes a threshold mean anything.
- **Rank slices by error mass, not error rate.** The worst class is often not the
  one with the most to gain, and the ceilings summing to the error budget is the
  check that keeps you honest.
- **Put the policy outside the model.** Some decisions must not be reachable by a
  confidence score, and that belongs in code with a test on it.
- **The reviewer queue is a labelling pipeline.** It is free, it is drawn from the
  hard part of the distribution, and it must never contaminate the eval set.

The full project, with the API, the FAISS index, the drift monitor and 29 tests,
is at `projects/computer-vision/visual-defect-triage/`.
""")

code("""
# The notebook is self-contained, so it can drift from src/. Re-derive the three
# headline numbers the repo's own pipeline reports, and assert them here.
assert 0.95 <= overall <= 0.99, f"overall accuracy moved: {overall:.3f}"
assert T > 1.2, f"the model should be overconfident, temperature was {T:.3f}"
crack = next(r for r in rows if r["slice"] == "hairline_crack")
passr = next(r for r in rows if r["slice"] == "pass")
assert crack["accuracy"] < passr["accuracy"], "hairline_crack should be the hard class"
assert crack["ceiling"] < passr["ceiling"], "the error-mass inversion is the whole lesson"
print(f"overall {overall:.3f} | temperature {T:.2f} | "
      f"crack {crack['accuracy']:.3f} (ceiling {crack['ceiling']:.4f}) "
      f"< pass ceiling {passr['ceiling']:.4f}")
print("all checks passed")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} with {len(C)} cells")

"""Generate synthetic embeddings so the whole pipeline runs with no downloads.

The point of the demo is the machinery around the model, calibration, the gate,
retrieval, slices and the review loop. None of that needs real images, so this
plants a dataset with known structure and lets the rest of the project be run and
tested offline.

Deliberately planted, because the report should be able to find them:
  - hairline_crack is rare (3%) and hard, which is the slice/ceiling lesson
  - the raw logits are overconfident, which is what temperature scaling fixes
"""
import json
from pathlib import Path

import numpy as np

from src.schemas import CLASSES

SEED = 13
DIM = 96                # smaller than a real 768-d ViT, same arithmetic
N = 12000

# Only this many images carry a label. The rest are unlabelled production
# traffic, which the index and the drift monitor still use. This is the
# labelling budget from the blog, and it is also what makes the model
# overconfident, because 3,000 labels in 96 dimensions overfits.
LABEL_BUDGET = 3000

EASY_SEP = 7.0          # how far apart the visually obvious classes sit
CRACK_DELTA = 3.6       # hairline_crack is a small offset from pass, not its own cluster
NOISE = 1.15

# share of traffic. Separation is handled below, because it is not uniform:
# six classes are visually obvious and one is not.
PROFILE = [
    ("pass", 0.62),
    ("scratch", 0.14),
    ("dent", 0.09),
    ("discolour", 0.06),
    ("contamination", 0.04),
    ("weld_void", 0.02),
    ("hairline_crack", 0.03),   # rare AND subtle, which is the whole lesson
]

OUT = Path("data")


def generate(n: int = N, seed: int = SEED):
    """Six well-separated classes, plus one that sits right next to pass.

    A hairline crack looks like a good part until you look properly, so modelling
    it as a small offset from the pass centre is both physically right and the
    thing that makes the slice/ceiling lesson visible.
    """
    rng = np.random.default_rng(seed)
    names = [p[0] for p in PROFILE]
    shares = np.array([p[1] for p in PROFILE], dtype="float64")
    shares = shares / shares.sum()

    centres = rng.normal(0, 1, size=(len(names), DIM))
    centres = centres / np.linalg.norm(centres, axis=1, keepdims=True) * EASY_SEP
    off = rng.normal(0, 1, DIM)
    centres[names.index("hairline_crack")] = (
        centres[names.index("pass")] + off / np.linalg.norm(off) * CRACK_DELTA)

    labels = rng.choice(len(names), size=n, p=shares)

    embeddings = np.zeros((n, DIM))
    rows = []
    for i, lab in enumerate(labels):
        name = names[lab]
        embeddings[i] = centres[lab] + rng.normal(0, NOISE, DIM)
        rows.append({
            "image_id": f"img_{i:05d}",
            "batch_id": f"batch_{i // 25:04d}",     # 25 photos per production batch
            "label": name,
            "line_id": f"line_{i % 4 + 1}",
            "shift": "night" if (i // 500) % 2 else "day",
            "path": f"data/images/img_{i:05d}.jpg",
        })
    return embeddings, np.array(labels), rows


def main() -> None:
    embeddings, labels, rows = generate()
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.save(OUT / "embeddings.npy", embeddings.astype("float32"))
    np.save(OUT / "labels.npy", labels)

    counts = {c: int((labels == i).sum()) for i, c in enumerate([p[0] for p in PROFILE])}
    print(f"wrote {len(rows)} rows to {OUT}/manifest.jsonl")
    print(f"class counts: {counts}")
    print(f"classes in schema: {len(CLASSES)}")


if __name__ == "__main__":
    main()

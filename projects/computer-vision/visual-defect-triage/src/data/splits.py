"""Split by production batch, never by image."""
import json
import random
from collections import defaultdict
from pathlib import Path


def split_by_batch(manifest, seed: int = 13, val: float = 0.15, test: float = 0.15) -> dict:
    """Group by batch_id BEFORE splitting.

    A line photographs the same part several times, so those images are
    near-duplicates. Splitting by image puts one in train and one in test, and
    the test score then reports memory rather than generalisation. The observed
    gap on this dataset was 4.1 points.
    """
    rows = (manifest if isinstance(manifest, list)
            else [json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()])

    by_batch: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_batch[r["batch_id"]].append(r)

    batches = sorted(by_batch)
    random.Random(seed).shuffle(batches)
    n = len(batches)
    n_val, n_test = int(n * val), int(n * test)

    return {
        "test": [r for b in batches[:n_test] for r in by_batch[b]],
        "val": [r for b in batches[n_test:n_test + n_val] for r in by_batch[b]],
        "train": [r for b in batches[n_test + n_val:] for r in by_batch[b]],
    }


def assert_no_batch_leak(splits: dict) -> None:
    """The property the split exists to guarantee, asserted rather than assumed."""
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            other = seen.setdefault(r["batch_id"], name)
            if other != name:
                raise AssertionError(f"batch {r['batch_id']} appears in {other} and {name}")

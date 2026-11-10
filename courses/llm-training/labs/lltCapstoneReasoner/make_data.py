"""Write the Countdown problems: data/train.jsonl (20,000, 3 and 4 numbers), data/train3.jsonl (20,000, 3 numbers, the
reference runs' training set) and data/test.jsonl (200: 100 with 3 numbers, 100 with 4). Same seed, same files. About 2-3 minutes on a laptop CPU."""
import hashlib
import json
from pathlib import Path

import countdown as cd

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)
train, test = cd.split_problems(seed=0, n_train=20000, n_test=200)
# The training set the reference runs use: 3-number problems only (Qwen3-0.6B-Base solves almost no 4-number
# problem at pass@8, so 4-number groups would all score zero). 4-number test problems then measure transfer.
held = {(tuple(sorted(p["nums"])), p["target"]) for p in test}
train3 = [p for p in cd.make_problems(20400, 5, sizes=(3,)) if (tuple(sorted(p["nums"])), p["target"]) not in held][:20000]
for name, rows in [("train.jsonl", train), ("test.jsonl", test), ("train3.jsonl", train3)]:
    text = "".join(json.dumps(p) + "\n" for p in rows)
    (out / name).write_text(text, encoding="utf-8")
    print(name, len(rows), "sha256", hashlib.sha256(text.encode()).hexdigest()[:16])

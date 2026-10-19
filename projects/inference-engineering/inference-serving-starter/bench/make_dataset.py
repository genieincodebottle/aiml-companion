"""Generate bench/data/length_pairs.jsonl and length_pairs_chat.jsonl. GIVEN. Deterministic (seed 7).

THE DATA IS SYNTHETIC. It is not a trace from a real product. The method, so you can say
so in your report:

* prompt_tokens: log-normal, median 512, sigma 0.8, clipped to [64, 3072]
* output_tokens: log-normal, median 128, sigma 0.7, clipped to [8, 512]
* 35 percent of rows start with one of 4 shared system prompts (prefix_group 0..3 of
  192, 320, 448 and 576 tokens). The shared part is counted inside prompt_tokens.
  Rows without a shared prefix have prefix_group null and prefix_tokens 0.

length_pairs_chat.jsonl is a second, short-context workload (short chat turns, no shared prefix):
* prompt_tokens: log-normal, median 96, sigma 0.5, clipped to [32, 400]
* output_tokens: log-normal, median 128, sigma 0.6, clipped to [16, 400]
The speculation experiment uses it on purpose: with short contexts the KV reads are small, so a
large batch becomes compute-bound and the speculation result can flip sign. With the long-context
file above, KV reads keep the step memory-bound and speculation keeps winning.

Real traffic has its own shape. Chat, RAG and agent loops differ a lot, and the shape
moves every number you will report. Your report's threats-to-validity section must say
that. Replace this file with your own length pairs if you have a real log.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "length_pairs.jsonl"
OUT_CHAT = DATA / "length_pairs_chat.jsonl"
PREFIX_TOKENS = [192, 320, 448, 576]
NL = chr(10)


def _write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=NL) as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + NL)
    print(f"wrote {len(rows)} rows to {path}")


def main(n: int = 2000, seed: int = 7) -> None:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = max(64, min(3072, int(round(rng.lognormvariate(math.log(512), 0.8)))))
        o = max(8, min(512, int(round(rng.lognormvariate(math.log(128), 0.7)))))
        group, ptoks = None, 0
        if rng.random() < 0.35:
            group = rng.randrange(4)
            ptoks = PREFIX_TOKENS[group]
            p = max(p, ptoks + 32)
        rows.append({"prompt_tokens": p, "output_tokens": o, "prefix_group": group, "prefix_tokens": ptoks})
    _write(OUT, rows)
    rng = random.Random(seed + 1)
    chat = []
    for _ in range(n):
        p = max(32, min(400, int(round(rng.lognormvariate(math.log(96), 0.5)))))
        o = max(16, min(400, int(round(rng.lognormvariate(math.log(128), 0.6)))))
        chat.append({"prompt_tokens": p, "output_tokens": o, "prefix_group": None, "prefix_tokens": 0})
    _write(OUT_CHAT, chat)


if __name__ == "__main__":
    main()

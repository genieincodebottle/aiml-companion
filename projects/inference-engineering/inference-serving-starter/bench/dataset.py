"""Workload: prompt and output lengths from a dataset file, turned into token-id prompts.

GIVEN: WorkItem, load_dataset, make_prompt_ids.
YOU:   sample_workload.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "length_pairs.jsonl"
VOCAB = 100_000   # below every common vocab size, so ids are valid on a real server too


@dataclass
class WorkItem:
    rid: int
    prompt_ids: list
    max_tokens: int
    prefix_group: int | None = None


def load_dataset(path=DEFAULT_DATASET) -> list:
    """Rows of {prompt_tokens, output_tokens, prefix_group, prefix_tokens}."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_prompt_ids(row: dict, rng: random.Random) -> list:
    """Token ids for one row. Rows in the same prefix_group share their first prefix_tokens ids
    exactly (that is what produces prefix-cache hits); the rest is fresh random ids."""
    ids: list = []
    g = row.get("prefix_group")
    if g is not None and row.get("prefix_tokens", 0) > 0:
        r = random.Random(7_919 * (g + 1))
        ids = [r.randrange(VOCAB) for _ in range(row["prefix_tokens"])]
    return ids + [rng.randrange(VOCAB) for _ in range(row["prompt_tokens"] - len(ids))]


def sample_workload(rows: list, n: int, seed: int) -> list:
    """YOU. Return `n` WorkItem objects drawn from `rows`.

    Contract (tests/test_bench.py):
      * Sample rows WITH replacement using random.Random(seed); same seed -> identical output.
      * rid runs 0..n-1 in order.
      * prompt_ids = make_prompt_ids(row, rng) with a rng seeded from `seed`, so
        len(prompt_ids) == row["prompt_tokens"] and rows of one prefix_group share a prefix.
      * max_tokens = row["output_tokens"], prefix_group = row["prefix_group"].
      * The empirical distribution of the sample must match the dataset (the test compares
        medians and the prefix share). Do not clip or reshape it.
    """
    raise NotImplementedError("bench/dataset.py: sample_workload")

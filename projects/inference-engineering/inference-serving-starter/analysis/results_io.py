"""Load result files written by bench.cli. GIVEN."""
from __future__ import annotations

import json
from pathlib import Path

from bench.records import RequestRecord


def load_results(path) -> tuple:
    """Returns (meta, levels). Each level is the JSON dict with "records" turned into
    RequestRecord objects (all of them, warm-up included; apply drop_warmup yourself)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    levels = []
    for lv in data["levels"]:
        lv = dict(lv)
        lv["records"] = [RequestRecord(**r) for r in lv["records"]]
        levels.append(lv)
    return data["meta"], levels

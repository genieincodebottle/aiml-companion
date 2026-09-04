"""Write each reviewer decision as a labelled example, tagged as mined.

sampling='mined' is the important field. These images were selected because the
model was unsure, so they are harder than production traffic. They may train.
They must never enter the evaluation set, or the test score drifts away from
reality and keeps drifting every cycle.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

TRAIN_ONLY = Path("data/reviewed.jsonl")


def record(image_id: str, ruling: str, reviewer: str, model_said: str,
           confidence: float, path: Path = TRAIN_ONLY) -> dict:
    row = {
        "image_id": image_id,
        "label": ruling,
        "reviewer": reviewer,
        "model_said": model_said,
        "model_confidence": confidence,
        "sampling": "mined",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_mined(path: Path = TRAIN_ONLY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def assert_not_in_evaluation(mined: list[dict], eval_ids: set[str]) -> None:
    """The flywheel's one silent failure, asserted."""
    leaked = {r["image_id"] for r in mined} & eval_ids
    if leaked:
        raise AssertionError(f"mined images leaked into the evaluation set: {sorted(leaked)[:5]}")

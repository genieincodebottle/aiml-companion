"""Folds the mined hard negatives back in and re-exports the engine."""
from __future__ import annotations

from pathlib import Path


def retrain(run_dir, load_site_data, load_mined, train_yolo, to_onnx,
            export_and_check, assert_parity):
    """Weights, engine, and parity test ship together or not at all.

    A new checkpoint with the old engine means the device runs the old
    model while every metric describes the new one, and nothing errors.

    The dependencies are arguments rather than imports so this sequence
    can be tested without a GPU. The sequence IS the thing worth having.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_site_data() + load_mined()
    weights = train_yolo(data)

    onnx = to_onnx(weights)
    engine = export_and_check(onnx, str(run_dir / "yolo.engine"))
    assert_parity(engine, weights)

    return {"run_dir": str(run_dir), "weights": weights, "engine": engine,
            "examples": len(data)}

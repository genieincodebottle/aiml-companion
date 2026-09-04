"""Head, temperature, and index are one deployable artefact.

The failure mode of shipping them separately is silent in every case. A new head
with the old temperature does not crash; it produces confidences that are wrong
by a consistent factor, so the gate routes a different share of traffic and
nobody can explain why review volume moved.
"""
from pathlib import Path

import numpy as np

from src.calibrate import fit_temperature
from src.models.backbone import EMBEDDING_VERSION
from src.retrieval.index import build
from src.train import train


def retrain(run_dir, embeddings: dict, labels: dict, ids: list[str], num_classes: int) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    head = train(embeddings["train"], labels["train"], num_classes)
    temperature = fit_temperature(head.forward(embeddings["val"]), labels["val"])
    index, index_ids = build(embeddings["all"], ids)

    head.save(run_dir / "head.npz")
    (run_dir / "manifest.json").write_text(
        f'{{"temperature": {temperature:.6f}, "embedding_version": "{EMBEDDING_VERSION}"}}',
        encoding="utf-8",
    )
    np.save(run_dir / "index_ids.npy", np.array(index_ids))
    return run_dir

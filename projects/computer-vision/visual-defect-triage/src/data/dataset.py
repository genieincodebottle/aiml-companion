"""Reads the manifest and returns tensors with the metadata slices need."""
import json
from pathlib import Path

from src.schemas import CLASSES, DefectClass


def read_manifest(path) -> list[dict]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_index(label: str) -> int:
    return CLASSES.index(DefectClass(label))


class DefectDataset:
    """torch.utils.data.Dataset when torch is installed; a plain sequence otherwise.

    line_id and shift ride along with every sample so evaluation can slice by
    them without joining back to a database. If the metadata is not in the batch,
    computing accuracy for the night shift needs a join, and the join never gets
    written.
    """

    def __init__(self, manifest, transform=None):
        self.rows = read_manifest(manifest)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        item = {
            "label": label_index(row["label"]),
            "line_id": row["line_id"],
            "shift": row["shift"],
            "image_id": row["image_id"],
            "batch_id": row["batch_id"],
        }
        if self.transform is not None:
            from PIL import Image

            img = Image.open(row["path"]).convert("RGB")
            item["pixel_values"] = self.transform(img)
        return item

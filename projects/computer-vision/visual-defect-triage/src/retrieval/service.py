"""Turns neighbour indices into the rulings a reviewer needs.

A stale index does not error. It returns neighbours that are simply wrong, and a
reviewer trusting them makes worse decisions than one shown nothing, so the
constructor refuses to build against a different embedding version.
"""
from src.errors import IndexStale
from src.models.backbone import EMBEDDING_VERSION
from src.schemas import DefectClass, Neighbour


class RetrievalService:
    def __init__(self, index, ids, meta, built_with: str, version: str = EMBEDDING_VERSION):
        if built_with != version:
            raise IndexStale(f"index built with {built_with}, model is {version}")
        self.index, self.ids, self.meta = index, ids, meta

    def neighbours(self, vec, k: int = 8) -> list[Neighbour]:
        sims, idx = self.index.search(vec, k)
        out = []
        for s, i in zip(sims, idx):
            image_id = self.ids[int(i)]
            m = self.meta[image_id]
            out.append(Neighbour(
                image_id=image_id,
                similarity=float(s),
                ruling=DefectClass(m["final_ruling"]),
                batch_id=m["batch_id"],
            ))
        return out

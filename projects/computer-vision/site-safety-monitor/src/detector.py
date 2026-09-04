"""Runs the exported engine and returns boxes in original coordinates."""
from __future__ import annotations

from src.budget import stage
from src.config import settings
from src.postprocess import nms
from src.preprocess import letterbox, unletterbox
from src.schemas import Box, PPEClass

#: Engine class index order. Fixed once, because an engine exported with a
#: different order is a silent relabelling of every prediction.
CLASS_ORDER = (
    PPEClass.PERSON,
    PPEClass.HELMET,
    PPEClass.NO_HELMET,
    PPEClass.VEST,
    PPEClass.NO_VEST,
)


class Detector:
    def __init__(self, engine, conf: float | None = None):
        self.engine = engine
        self.conf = settings.conf_threshold if conf is None else conf

    def __call__(self, frame) -> list[Box]:
        with stage("letterbox"):
            img, meta = letterbox(frame)

        with stage("inference"):
            raw = self.engine.infer(img)          # (N, 5 + num_classes)

        with stage("nms"):
            kept = nms(raw, self.conf, settings.iou_threshold,
                       settings.max_detections)

        out = []
        for d in kept:
            x1, y1, x2, y2 = unletterbox(d[:4], meta)
            out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2,
                           cls=CLASS_ORDER[int(d[5])], conf=float(d[4])))
        return out

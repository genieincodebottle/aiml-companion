"""The endpoint. One forward pass, then classify, retrieve, and route from it.

Read the order in triage(): one backbone call produces vec, and the classifier,
the gate, and retrieval all consume it. The obvious way to add retrieval later is
to call the model again inside the retrieval service, which doubles the GPU cost
for a feature that needed none.
"""
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from api.deps import get_pipeline
from src.gate import route
from src.models.backbone import EMBEDDING_VERSION
from src.schemas import CLASSES, Prediction

app = FastAPI(title="Visual Defect Triage")

_backbone = None
_transform = None


def _lazy_model():
    """Imported on first use so the API module can be imported without torch."""
    global _backbone, _transform
    if _backbone is None:
        from src.data.transforms import eval_transform
        from src.models.backbone import Backbone

        _backbone = Backbone()
        _transform = eval_transform()
    return _backbone, _transform


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_version": EMBEDDING_VERSION}


@app.post("/triage", response_model=Prediction)
async def triage(image_id: str, file: UploadFile = File(...)) -> Prediction:
    head, retrieval, temperature = get_pipeline()
    backbone, transform = _lazy_model()

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, "unreadable image") from exc

    x = transform(img).unsqueeze(0)
    vec = backbone(x).cpu().numpy()[0]                 # the ONLY forward pass

    probs = head.probabilities(vec, temperature)[0]
    idx = int(np.argmax(probs))
    predicted = CLASSES[idx]
    confidence = float(probs[idx])

    return Prediction(
        image_id=image_id,
        predicted=predicted,
        confidence=confidence,
        route=route(predicted, confidence),
        neighbours=retrieval.neighbours(vec),
        embedding_version=EMBEDDING_VERSION,
    )

"""The prediction, the route, and the review record. Every stage speaks these."""
from enum import Enum

from pydantic import BaseModel, Field


class DefectClass(str, Enum):
    PASS = "pass"
    SCRATCH = "scratch"
    DENT = "dent"
    DISCOLOUR = "discolour"
    CONTAMINATION = "contamination"
    WELD_VOID = "weld_void"
    HAIRLINE_CRACK = "hairline_crack"


CLASSES = list(DefectClass)


class Route(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    AUTO_REJECT = "auto_reject"
    REVIEW = "review"


class Neighbour(BaseModel):
    image_id: str
    similarity: float
    ruling: DefectClass
    batch_id: str


class Prediction(BaseModel):
    """confidence is bounded at the type level, so an uncalibrated logit cannot
    reach the gate pretending to be a probability."""

    image_id: str
    predicted: DefectClass
    confidence: float = Field(ge=0.0, le=1.0)
    route: Route
    neighbours: list[Neighbour] = []
    # The field people leave out and regret. After a backbone change every vector
    # in the index came from the old one, and comparing across versions returns
    # neighbours that are quietly wrong.
    embedding_version: str

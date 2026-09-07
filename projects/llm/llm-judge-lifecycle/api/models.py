"""Request and response shapes.

Responses are deliberately loose (``dict[str, Any]``) because the services
already produce the exact payload the UI and the CLI render, and re-declaring
every field here would mean two definitions of one result that drift apart.
Requests are strict, because they arrive from outside.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TuneRequest(BaseModel):
    criterion: str
    #: False reproduces the paper's vanilla arm: label mismatches only.
    reasoning_alignment: bool = True


class EvalRequest(BaseModel):
    criterion: str
    split: Literal["train", "validation", "test"] = "test"
    staged: bool = False


class ServeRequest(BaseModel):
    record_id: str
    # Bounded here as well as in config. The retry budget is the one input a
    # caller can use to multiply the cost of a single request, so it is capped
    # at the transport boundary rather than trusted.
    max_retries: int | None = Field(default=None, ge=0, le=12)


class Result(BaseModel):
    ok: bool = True
    data: dict[str, Any]
    #: Which model produced this, whether the run was fully offline, and whether
    #: the generator and judge were the same model. Returned on every response
    #: so a number can never travel without the caveats attached to it.
    provenance: dict[str, Any]

"""HTTP client for the UI. The only way the frontend reaches the system.

No business logic here and none in the Streamlit app either. Importing ``src``
from the UI would let it bypass every guardrail the API enforces - the budget
cap, the exception mapping, the provenance stamp - and a control the frontend
can skip is a control anyone can skip with curl.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = os.environ.get("JUDGE_API_URL", "http://localhost:8000")

# Generous, and deliberately so. A tuning run scores the whole training split
# once per iteration against a hosted model; a 30-second default would time out
# mid-run and leave the user unable to tell a slow loop from a broken one.
TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class ApiError(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            response = client.request(method, path, **kwargs)
    except httpx.RequestError as exc:
        raise ApiError(
            f"cannot reach the API at {BASE_URL}. Start it with:\n"
            "    uvicorn api.main:app --port 8000\n"
            f"({type(exc).__name__})"
        ) from exc

    if response.status_code >= 400:
        # Surface the service's own message. The API already explains WHY - an
        # unknown criterion lists the available ones, a budget stop reports the
        # spend per role - and replacing that with "request failed" throws away
        # the only useful part.
        try:
            detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text
        raise ApiError(f"{response.status_code}: {detail}")

    return response.json()


def health() -> dict[str, Any]:
    return _request("GET", "/api/health")


def benchmark_report() -> dict[str, Any]:
    return _request("GET", "/api/benchmark/report")


def evaluate(criterion: str, split: str, staged: bool = False) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/benchmark/evaluate",
        json={"criterion": criterion, "split": split, "staged": staged},
    )


def tune(criterion: str, reasoning_alignment: bool = True) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/tuning/tune",
        json={"criterion": criterion, "reasoning_alignment": reasoning_alignment},
    )


def ablation(criterion: str) -> dict[str, Any]:
    return _request("POST", f"/api/tuning/ablation/{criterion}")


def promote(criterion: str) -> dict[str, Any]:
    return _request("POST", f"/api/tuning/promote/{criterion}")


def serve(record_id: str, max_retries: int | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"record_id": record_id}
    if max_retries is not None:
        body["max_retries"] = max_retries
    return _request("POST", "/api/serving/serve", json=body)


def retry_curve(max_k: int = 6) -> dict[str, Any]:
    return _request("GET", "/api/serving/retry-curve", params={"max_k": max_k})


def monitoring_weeks() -> dict[str, Any]:
    return _request("GET", "/api/monitoring/weeks")


def monitoring_check(week: int) -> dict[str, Any]:
    return _request("GET", f"/api/monitoring/check/{week}")

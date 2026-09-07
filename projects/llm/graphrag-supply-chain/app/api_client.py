"""HTTP client for the backend. The ONLY place the UI talks to anything.

Every function here is a thin wrapper over one endpoint. There is no retrieval
logic, no Cypher, no prompt, no guardrail and no scoring in this file or
anywhere else under `app/` - all of it lives behind the API.

That separation is not tidiness. A guardrail enforced in a frontend is a
guardrail anyone can skip with curl, and business logic duplicated between a UI
and a backend drifts apart within a week. The UI's job is to render what the
API returns and to make the system legible; the API's job is to be correct.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Ingestion and evaluation are polled, not awaited, so no call here should ever
# need minutes. A generous per-request timeout still bounds a hung backend.
_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)


class ApiError(RuntimeError):
    """A backend error, carrying the structured detail the API returned.

    Guardrail refusals arrive here as 400s with a `kind`, which is what lets the
    UI show "this was blocked because it looked like prompt injection" instead
    of a generic failure. Losing that structure is how a well-designed refusal
    turns into an unexplained error message.
    """

    def __init__(self, message: str, *, status: int = 0,
                 kind: str = "", detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.detail = detail


class ApiClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY") or ""

    # ------------------------------------------------------------------ core
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                response = client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                f"Cannot reach the API at {self.base_url}.\n\n"
                "Start it in a second terminal:\n"
                "    python run.py api\n\n"
                "The UI holds no business logic by design, so it cannot answer "
                "anything without the backend.",
                kind="connection",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ApiError(
                "The API took too long to respond. Long jobs (ingestion, "
                "evaluation) run in the background and are polled - if this "
                "happened on a question, the model provider is likely slow.",
                kind="timeout",
            ) from exc

        if response.status_code >= 400:
            raise _to_error(response)
        return response.json()

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params or None)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=payload or {})

    # ------------------------------------------------------------------ meta
    def health(self) -> dict[str, Any]:
        return self.get("/health")

    def reachable(self) -> tuple[bool, str]:
        try:
            health = self.health()
            return health.get("status") == "ok", ""
        except ApiError as exc:
            return False, str(exc)

    # ----------------------------------------------------------------- graph
    def census(self) -> dict[str, Any]:
        return self.get("/api/graph/census")

    def schema(self) -> dict[str, Any]:
        return self.get("/api/graph/schema")

    def entities(self, entity_type: str | None = None, search: str | None = None,
                 limit: int = 500) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        if search:
            params["search"] = search
        return self.get("/api/graph/entities", **params)

    def entity(self, key: str) -> dict[str, Any]:
        return self.get(f"/api/graph/entity/{key}")

    def subgraph(self, keys: list[str], hops: int = 2,
                 limit: int = 60) -> dict[str, Any]:
        # httpx encodes a list value as repeated query parameters, which is what
        # FastAPI's `Query(...)` list binding expects.
        return self.get("/api/graph/subgraph", key=keys, hops=hops, limit=limit)

    def cookbook(self) -> list[dict[str, Any]]:
        return self.get("/api/graph/cookbook")

    def run_cypher(self, cypher: str,
                   parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.post("/api/graph/cypher",
                         {"cypher": cypher, "parameters": parameters or {}})

    # ------------------------------------------------------------------- ask
    def strategies(self) -> list[dict[str, str]]:
        return self.get("/api/strategies")

    def ask(self, question: str, strategy: str = "hybrid",
            include_trace: bool = True) -> dict[str, Any]:
        return self.post("/api/ask", {"question": question, "strategy": strategy,
                                      "include_trace": include_trace})

    def compare(self, question: str, strategies: list[str]) -> dict[str, Any]:
        return self.post("/api/compare",
                         {"question": question, "strategies": strategies})

    # ------------------------------------------------------------------ jobs
    def start_ingest(self, reset: bool = True) -> dict[str, Any]:
        return self.post("/api/ingest", {"reset": reset})

    def ingest_status(self) -> dict[str, Any]:
        return self.get("/api/ingest/status")

    def start_eval(self, strategies: list[str] | None = None,
                   question_ids: list[str] | None = None,
                   judge: bool = False) -> dict[str, Any]:
        return self.post("/api/eval", {"strategies": strategies,
                                       "question_ids": question_ids,
                                       "judge": judge})

    def eval_status(self) -> dict[str, Any]:
        return self.get("/api/eval/status")

    def golden_questions(self) -> list[dict[str, Any]]:
        return self.get("/api/eval/questions")

    # ------------------------------------------------------------ guardrails
    def guardrail_config(self) -> dict[str, Any]:
        return self.get("/api/guardrails/config")

    def scan(self, text: str, as_document: bool = True) -> dict[str, Any]:
        return self.post("/api/guardrails/scan",
                         {"text": text, "as_document": as_document})

    def audit(self, limit: int = 100, event: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if event:
            params["event"] = event
        return self.get("/api/guardrails/audit", **params)

    def adversarial_sample(self) -> dict[str, Any]:
        return self.get("/api/guardrails/adversarial-sample")


def _to_error(response: httpx.Response) -> ApiError:
    try:
        payload = response.json()
    except ValueError:
        return ApiError(response.text[:400] or f"HTTP {response.status_code}",
                        status=response.status_code)

    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        return ApiError(detail.get("message", str(detail)),
                        status=response.status_code,
                        kind=detail.get("kind", ""), detail=detail.get("detail"))
    if isinstance(detail, list):
        # FastAPI validation errors. Rendered as the field plus the reason,
        # because "value is not a valid ..." with no field name is useless.
        parts = [
            f"{'.'.join(str(x) for x in item.get('loc', [])[1:])}: {item.get('msg', '')}"
            for item in detail
        ]
        return ApiError("; ".join(parts) or "invalid request",
                        status=response.status_code, kind="validation")
    return ApiError(str(detail), status=response.status_code)

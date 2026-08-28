"""Northwind Freight legacy dispatch API, 2019 vintage.

Run:  uvicorn customer.legacy_api.server:app --port 8000

You did not build this and you cannot change it. Integrate against it as it
is. Every quirk below is real behaviour copied from systems of this era, and
every one of them is beatable with a well written client.

QUIRKS, in full, because training material should be discoverable by reading
and not only by suffering:

 1. GET /api/v1/shipments paginates, but the cursor key is not stable.
    Even numbered pages return "next_offset". Odd numbered pages return
    "nextPage". A client that reads only one of them stops early and silently
    loses half the dataset.

 2. The last page does not return a null cursor. It returns an empty "data"
    list with a cursor key still present. Termination must be detected from
    the emptiness of the page, not from the cursor.

 3. GET /api/v1/shipment/{id} returns XML with content type application/xml,
    while every other endpoint returns JSON. A client that calls .json() on it
    raises.

 4. GET /api/v1/drivers returns HTTP 200 with a body of {"error": "..."} on
    roughly one call in eight. It is driven by an in process call counter, not
    by randomness, so it is reproducible: every 8th call to this endpoint
    fails. Status code checking alone will not catch it.

 5. POST /api/v1/shipment/{id}/reschedule is the only write endpoint.
      - Requires header X-Api-Key. The accepted value is "northwind-dev-key".
        A missing key returns 401, a wrong key returns 403. Both bodies are
        JSON but the field names differ between the two.
      - It is NOT idempotent. Calling it twice creates two reschedule records
        unless the caller sends an Idempotency-Key header. With that header,
        a repeat call returns the original record and "duplicate": true.

 6. Rate limiting is global and counted, not timed. After 20 requests to any
    /api/v1/* endpoint the API returns 429 with a Retry-After header. The
    window resets after a further 5 rejected calls, which is the behaviour of
    the original leaky bucket and is not documented anywhere on the customer
    side. It is a counter so tests are reproducible.

 7. GET /health is outside the rate limiter and always returns 200. It is the
    only endpoint you can trust unconditionally.

 8. POST /api/v1/_reset clears all in process state (counters, reschedules).
    It exists because the 2019 team needed it for their own tests. It is not
    in the customer's own documentation. Use it between test cases.

All state is a module level dict. There is no database. Restarting the
process resets everything.
"""

from __future__ import annotations

from typing import Any, Dict, List
from xml.sax.saxutils import escape

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

API_KEY = "northwind-dev-key"
PAGE_SIZE = 25
TOTAL_SHIPMENTS = 137
RATE_LIMIT = 20
RATE_LIMIT_COOLDOWN = 5
DRIVER_FAIL_EVERY = 8

app = FastAPI(title="Northwind Dispatch API", version="1.0.0")

STATE: Dict[str, Any] = {
    "request_count": 0,
    "rejected_count": 0,
    "driver_calls": 0,
    "reschedules": [],
    "idempotency": {},
}

DRIVERS = [
    {"driver_id": "D-1001", "name": "R. Vermeulen", "depot": "Rotterdam", "vehicle": "7.5t"},
    {"driver_id": "D-1002", "name": "K. Osei", "depot": "Antwerp", "vehicle": "artic"},
    {"driver_id": "D-1003", "name": "M. Lindqvist", "depot": "Hamburg", "vehicle": "van"},
    {"driver_id": "D-1004", "name": "P. Almeida", "depot": "Porto", "vehicle": "7.5t"},
    {"driver_id": "D-1005", "name": "S. Novak", "depot": "Gdansk", "vehicle": "artic"},
]

STATUSES = ["delivered", "in_transit", "failed", "returned"]


def _shipment(index: int) -> Dict[str, Any]:
    """Synthesise a stable shipment record from its index."""
    return {
        "shipment_id": "SH25%05d" % (10000 + index),
        "cust_id": "%07d" % (4000 + (index * 7) % 1500),
        "status": STATUSES[index % len(STATUSES)],
        "origin_depot": DRIVERS[index % len(DRIVERS)]["depot"],
        "dest_city": ["Leeds", "Lyon", "Turin", "Utrecht", "Malmo"][index % 5],
        "attempts": index % 4,
        "weight_kg": round(1.5 + (index * 3.7) % 88.0, 2),
    }


def _rate_limited() -> Response | None:
    """Global counted rate limiter. Returns a 429 response or None."""
    STATE["request_count"] += 1
    if STATE["request_count"] > RATE_LIMIT:
        STATE["rejected_count"] += 1
        if STATE["rejected_count"] >= RATE_LIMIT_COOLDOWN:
            STATE["request_count"] = 0
            STATE["rejected_count"] = 0
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "code": "RL-002"},
            headers={"Retry-After": "3"},
        )
    return None


@app.get("/health")
def health() -> Dict[str, str]:
    """Always works. Outside the rate limiter on purpose."""
    return {"status": "ok", "service": "northwind-dispatch", "version": "1.0.0"}


@app.post("/api/v1/_reset")
def reset() -> Dict[str, str]:
    """Undocumented on the customer side. Clears all in process state."""
    STATE["request_count"] = 0
    STATE["rejected_count"] = 0
    STATE["driver_calls"] = 0
    STATE["reschedules"] = []
    STATE["idempotency"] = {}
    return {"status": "reset"}


@app.get("/api/v1/shipments")
def list_shipments(offset: int = 0, limit: int = PAGE_SIZE) -> Any:
    """Paginated list. The cursor key alternates by page number.

    Even page -> "next_offset". Odd page -> "nextPage". Last page returns an
    empty data list rather than a null cursor.
    """
    limited = _rate_limited()
    if limited is not None:
        return limited

    limit = max(1, min(limit, PAGE_SIZE))
    rows: List[Dict[str, Any]] = [
        _shipment(i) for i in range(offset, min(offset + limit, TOTAL_SHIPMENTS))
    ]
    page_number = offset // limit if limit else 0

    body: Dict[str, Any] = {"data": rows, "count": len(rows), "total": TOTAL_SHIPMENTS}
    next_offset = offset + limit
    if page_number % 2 == 0:
        body["next_offset"] = next_offset
    else:
        body["nextPage"] = next_offset
    return body


@app.get("/api/v1/shipment/{shipment_id}")
def get_shipment(shipment_id: str) -> Response:
    """Returns XML. Everything else on this API returns JSON."""
    limited = _rate_limited()
    if limited is not None:
        return limited

    try:
        index = int(shipment_id.replace("SH25", "")) - 10000
    except ValueError:
        index = -1

    if index < 0 or index >= TOTAL_SHIPMENTS:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<error><code>NOT_FOUND</code>"
            "<message>No such shipment %s</message></error>" % escape(shipment_id)
        )
        return Response(content=xml, media_type="application/xml", status_code=404)

    row = _shipment(index)
    parts = "".join("<%s>%s</%s>" % (k, escape(str(v)), k) for k, v in row.items())
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<shipment>%s</shipment>' % parts
    return Response(content=xml, media_type="application/xml")


@app.get("/api/v1/drivers")
def list_drivers() -> Any:
    """Returns 200 with an error body on every 8th call. Counted, not random."""
    limited = _rate_limited()
    if limited is not None:
        return limited

    STATE["driver_calls"] += 1
    if STATE["driver_calls"] % DRIVER_FAIL_EVERY == 0:
        # HTTP 200. The failure is in the body only.
        return {"error": "Upstream driver roster unavailable, try again", "code": "DR-511"}

    return {"data": DRIVERS, "count": len(DRIVERS)}


@app.post("/api/v1/shipment/{shipment_id}/reschedule")
async def reschedule(
    shipment_id: str,
    request: Request,
    x_api_key: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None),
) -> Any:
    """The only write endpoint. Auth required, non-idempotent without a key."""
    limited = _rate_limited()
    if limited is not None:
        return limited

    if x_api_key is None:
        # Note the field name. It is not the same as the 403 body below.
        return JSONResponse(status_code=401, content={"error": "Missing X-Api-Key header"})
    if x_api_key != API_KEY:
        return JSONResponse(status_code=403, content={"message": "Invalid api key", "code": "AUTH-9"})

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    new_date = payload.get("new_date", "unspecified")

    if idempotency_key is not None and idempotency_key in STATE["idempotency"]:
        original = STATE["idempotency"][idempotency_key]
        return {"reschedule": original, "duplicate": True}

    record = {
        "reschedule_id": "RS-%05d" % (1000 + len(STATE["reschedules"])),
        "shipment_id": shipment_id,
        "new_date": new_date,
        "requested_by": payload.get("requested_by", "unknown"),
    }
    STATE["reschedules"].append(record)
    if idempotency_key is not None:
        STATE["idempotency"][idempotency_key] = record

    return {"reschedule": record, "duplicate": False}


@app.get("/api/v1/reschedules")
def list_reschedules() -> Any:
    """Read back what the write endpoint created. Useful for proving duplicates."""
    limited = _rate_limited()
    if limited is not None:
        return limited
    return {"data": STATE["reschedules"], "count": len(STATE["reschedules"])}

"""The tools the sub-agents actually call.

Every one of these hits the seeded SQLite database. Nothing here returns a
canned dict, because the failure modes the project is about only show up when
the tool can genuinely be slow, genuinely return nothing, or genuinely leave a
side effect behind that has to be undone.

Two rules the post insists on and this file follows:

  - the tenant id is part of the query, never a filter applied afterwards
  - a tool that runs out of time returns a timeout the agent can reason about,
    not an exception that kills the branch
"""

from __future__ import annotations

import asyncio
from typing import Any

from .db import connect


class ToolTimeout(Exception):
    pass


async def _maybe_delay(name: str, switches: Any) -> None:
    """The injected latency lives here rather than inside each tool, so the
    slow-tool toggle can target any tool by name."""
    if switches and switches.slow_tool_seconds > 0 and switches.slow_tool_target == name:
        await asyncio.sleep(switches.slow_tool_seconds)


async def get_customer(tenant_id: str, customer_id: str, switches: Any = None) -> dict[str, Any]:
    await _maybe_delay("get_customer", switches)
    conn = connect()
    try:
        row = conn.execute(
            # tenant_id is in the WHERE clause, not applied to the result set.
            "SELECT * FROM customers WHERE tenant_id = ? AND customer_id = ?",
            (tenant_id, customer_id),
        ).fetchone()
        return dict(row) if row else {"error": "not_found", "customer_id": customer_id}
    finally:
        conn.close()


async def get_order(tenant_id: str, order_id: str, switches: Any = None) -> dict[str, Any]:
    await _maybe_delay("get_order", switches)
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE tenant_id = ? AND order_id = ?",
            (tenant_id, order_id),
        ).fetchone()
        return dict(row) if row else {"error": "not_found", "order_id": order_id}
    finally:
        conn.close()


async def list_orders(tenant_id: str, customer_id: str, switches: Any = None) -> list[dict[str, Any]]:
    await _maybe_delay("list_orders", switches)
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM orders WHERE tenant_id = ? AND customer_id = ? ORDER BY placed_at DESC",
            (tenant_id, customer_id),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


async def get_shipment(tenant_id: str, order_id: str, switches: Any = None) -> dict[str, Any]:
    """The default target of the slow-tool toggle. Twelve seconds here is what
    makes the fan-out inherit its slowest agent."""
    await _maybe_delay("get_shipment", switches)
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM shipments WHERE tenant_id = ? AND order_id = ?",
            (tenant_id, order_id),
        ).fetchone()
        return dict(row) if row else {"error": "not_found", "order_id": order_id}
    finally:
        conn.close()


async def track_parcel(tenant_id: str, tracking: str, switches: Any = None) -> dict[str, Any]:
    await _maybe_delay("track_parcel", switches)
    conn = connect()
    try:
        row = conn.execute(
            "SELECT courier, tracking, status, eta, last_scan_at FROM shipments "
            "WHERE tenant_id = ? AND tracking = ?",
            (tenant_id, tracking),
        ).fetchone()
        return dict(row) if row else {"error": "unknown_tracking", "tracking": tracking}
    finally:
        conn.close()


async def call_with_timeout(coro: Any, timeout_s: float, tool_name: str) -> dict[str, Any]:
    """Wrap any tool in the budget the agent has left. The agent sees a result
    it can reason about either way, which is the whole point: a timeout is data,
    not a crash."""
    try:
        return await asyncio.wait_for(coro, timeout=max(timeout_s, 0.01))
    except asyncio.TimeoutError:
        return {
            "error": "tool_timeout",
            "tool": tool_name,
            "budget_s": round(timeout_s, 2),
            "note": "no result inside the budget this agent had left",
        }


TOOL_NAMES = [
    "get_customer",
    "get_order",
    "list_orders",
    "get_shipment",
    "track_parcel",
    "search_policy",
]

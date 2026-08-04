"""Sagas: a sequence of changes across several systems, each paired with its undo.

The post's scenario, implemented literally: a plan books a courier, charges a
fee, and updates the order. The third step fails. Two changes are now live in
two systems and there is no transaction across them to roll back.

The rule this file follows is the one that is easy to write down and easy to
skip: build each undo in the same change as the step it undoes. Here that means
a step cannot be registered without its compensator, because `Step` requires
both. Left for later it never gets built, and you find that out during the first
failure rather than before it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .db import connect

Action = Callable[[], Awaitable[dict[str, Any]]]


@dataclass
class Step:
    name: str
    system: str
    do: Action
    # Not optional. A step without an undo cannot be added to a saga.
    undo: Action


@dataclass
class StepResult:
    name: str
    system: str
    direction: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)


def _log(request_id: str, step: str, direction: str, payload: dict[str, Any]) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO saga_log (request_id, step, direction, payload, at) VALUES (?,?,?,?,?)",
            (
                request_id,
                step,
                direction,
                json.dumps(payload),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


class Saga:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.steps: list[Step] = []
        self.results: list[StepResult] = []
        self.rolled_back = False

    def add(self, step: Step) -> "Saga":
        self.steps.append(step)
        return self

    async def run(self) -> list[StepResult]:
        """Run forward. On the first failure, run the undos of everything that
        already succeeded, in reverse order, and stop."""
        completed: list[Step] = []

        for step in self.steps:
            try:
                payload = await step.do()
            except Exception as exc:  # noqa: BLE001 - a failed step is expected here
                payload = {"error": str(exc)}

            failed = "error" in payload
            self.results.append(
                StepResult(
                    step.name,
                    step.system,
                    "forward",
                    "error" if failed else "ok",
                    payload,
                )
            )
            _log(self.request_id, step.name, "forward", payload)

            if failed:
                await self._compensate(completed)
                return self.results
            completed.append(step)

        return self.results

    async def _compensate(self, completed: list[Step]) -> None:
        self.rolled_back = True
        # Reverse order. Refund the fee before cancelling the courier booking,
        # because the fee was charged after the booking was made.
        for step in reversed(completed):
            try:
                payload = await step.undo()
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                payload = {"error": str(exc)}
                # A failed compensator is the genuinely bad case. It needs a
                # human, and it needs to be loud.
                status = "error"
            self.results.append(
                StepResult(f"undo:{step.name}", step.system, "compensate", status, payload)
            )
            _log(self.request_id, f"undo:{step.name}", "compensate", payload)


def build_return_booking_saga(
    *, request_id: str, tenant_id: str, order_id: str, fail_order_update: bool
) -> Saga:
    """The three-step change from the post, with its three undos."""
    state: dict[str, Any] = {}

    async def book_courier() -> dict[str, Any]:
        state["booking_ref"] = f"BK-{order_id[-4:]}-01"
        return {"booking_ref": state["booking_ref"], "courier": "Meridian Express"}

    async def cancel_courier() -> dict[str, Any]:
        return {"cancelled": state.get("booking_ref")}

    async def charge_fee() -> dict[str, Any]:
        state["charge_id"] = f"CH-{order_id[-4:]}-01"
        return {"charge_id": state["charge_id"], "amount_usd": 8.00}

    async def refund_fee() -> dict[str, Any]:
        return {"refunded_charge": state.get("charge_id"), "amount_usd": 8.00}

    async def update_order() -> dict[str, Any]:
        if fail_order_update:
            # The realistic version of this failure: the write goes to a system
            # that is briefly unavailable, after two other systems have already
            # been changed.
            return {"error": "order_service_unavailable", "retryable": True}
        conn = connect()
        try:
            conn.execute(
                "UPDATE orders SET status = 'return_booked' WHERE tenant_id = ? AND order_id = ?",
                (tenant_id, order_id),
            )
            conn.commit()
        finally:
            conn.close()
        return {"order_id": order_id, "status": "return_booked"}

    async def revert_order() -> dict[str, Any]:
        conn = connect()
        try:
            conn.execute(
                "UPDATE orders SET status = 'delivered' WHERE tenant_id = ? AND order_id = ?",
                (tenant_id, order_id),
            )
            conn.commit()
        finally:
            conn.close()
        return {"order_id": order_id, "status": "delivered"}

    saga = Saga(request_id)
    saga.add(Step("book_courier", "courier-api", book_courier, cancel_courier))
    saga.add(Step("charge_fee", "payments", charge_fee, refund_fee))
    saga.add(Step("update_order", "order-service", update_order, revert_order))
    return saga

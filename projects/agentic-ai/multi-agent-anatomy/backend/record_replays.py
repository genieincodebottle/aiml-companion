"""Record the shipped traces.

Run this to regenerate backend/replay/*.json. It uses the replay client, so it
needs no key either: the recorded traces are real runs of the real pipeline over
the real seeded database, with each failure toggle set.

    python record_replays.py

The recorded set is what somebody who clones this repo sees first. It is built
here rather than at the end of the project on purpose, because it is the thing
that decides how many readers ever run it.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# Force replay mode before app.config is imported, so a developer who happens to
# have a key in .env does not silently record live traces: they would be
# nondeterministic, they would cost money, and they would defeat the purpose of
# a recorded set that anybody can reproduce.
os.environ["REPLAY_ONLY"] = "true"

from app.failures import FailureSwitches
from app.pipeline import cost_breakdown, run_request
from app.tools.db import init_db

OUT = Path(__file__).resolve().parent / "replay"

QUESTION = "Where is my order ORD-4412 and can I still return it?"

SCENARIOS = [
    {
        "id": "01-happy-path",
        "title": "1. The happy path",
        "teaches": "Eight stages, five agents, six model calls. Read the waterfall once here so the broken ones are legible.",
        "failures": {},
    },
    {
        "id": "02-shipping-agent-killed",
        "title": "2. Partial failure: shipping agent killed",
        "teaches": "The answer still goes out, built from the two branches that returned, with the gap stated instead of filled.",
        "failures": {"kill_shipping_agent": True},
    },
    {
        "id": "03-slow-tool",
        "title": "3. The fan-out inherits its slowest agent",
        "teaches": "12s of tool latency against an 8s per-agent timeout. The branch is cut, the request is not.",
        "failures": {"slow_tool_seconds": 12.0, "slow_tool_target": "get_shipment"},
    },
    {
        "id": "04-saga-rollback",
        "title": "4. Saga rollback across three systems",
        "teaches": "Book the courier, charge the fee, update the order. Step 3 fails and the undos run in reverse.",
        "failures": {"fail_order_update": True},
        "question": "I want to return ORD-4412, please book the return and refund it.",
    },
    {
        "id": "05-green-and-wrong",
        "title": "5. Every span green, the answer wrong",
        "teaches": "A superseded policy passage. Retrieval succeeded, the citation is real, nothing is red, and the refund window in the answer is 14 days instead of 30.",
        "failures": {"corrupt_passage": True},
    },
    {
        "id": "06-cache-off",
        "title": "6. Prompt caching off",
        "teaches": "The same request with every stable prefix re-billed at full input price. Compare the total against scenario 1.",
        "failures": {"disable_prompt_cache": True},
    },
    {
        "id": "07-cross-tenant-attempt",
        "title": "7. Cross-tenant retrieval attempt",
        "teaches": "The tenant id is inside the query. Another retailer's passage is never loaded, not loaded and then dropped.",
        "failures": {"attempt_cross_tenant": True},
    },
]


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        # Two runs per scenario, and the second one is what gets recorded.
        #
        # The first run pays full price for every stable prefix, because a cache
        # is empty until something fills it. Production requests almost never
        # look like that, so recording the cold run would overstate the bill of
        # every scenario and flatten the difference the caching toggle is there
        # to show. The reset makes the pair reproducible.
        # Reseed between scenarios. The saga scenario writes to the orders
        # table, and a recorded trace that inherits another scenario's side
        # effects is not a recording of that scenario.
        init_db(force=True)
        switches = FailureSwitches.from_dict(sc["failures"])
        await run_request(sc.get("question", QUESTION), switches=switches, reset_cache=True)
        result = await run_request(sc.get("question", QUESTION), switches=switches)
        result["cost_by_model"] = cost_breakdown(result["trace"])
        payload = {
            "id": sc["id"],
            "title": sc["title"],
            "teaches": sc["teaches"],
            "failures": sc["failures"],
            "question": sc.get("question", QUESTION),
            **result,
        }
        (OUT / f"{sc['id']}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"recorded {sc['id']}  cost=${result['trace']['summary']['total_cost_usd']:.5f}")


if __name__ == "__main__":
    asyncio.run(main())

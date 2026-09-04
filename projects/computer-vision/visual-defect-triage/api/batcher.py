"""Collect requests for a few milliseconds and run them as one batch.

Measured on this service, 4.0 ms fixed per batch plus 6.0 ms per image. Batch 1
is 100 images a second, batch 16 is 160, and the ceiling is 166.67, so batch 16
already takes 96 percent of what batching can ever give.

The timeout is what makes this safe under light load. Without it a half-full
batch waits for a request that never arrives.
"""
import asyncio

FIXED_MS = 4.0
PER_IMAGE_MS = 6.0


def throughput(batch_size: int) -> float:
    """Images a second at a given batch size. The curve, from two measurements."""
    per_image = (FIXED_MS + PER_IMAGE_MS * batch_size) / batch_size
    return 1000.0 / per_image


def ceiling() -> float:
    """As the batch grows the fixed term vanishes, so this is never reached."""
    return 1000.0 / PER_IMAGE_MS


class MicroBatcher:
    def __init__(self, run_batch, max_batch: int = 16, max_wait_ms: float = 20.0):
        self.run_batch = run_batch
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()

    async def run(self) -> None:
        while True:
            items = [await self.queue.get()]
            deadline = asyncio.get_running_loop().time() + self.max_wait
            while len(items) < self.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break

            for item, row in zip(items, self.run_batch([i["vector"] for i in items])):
                item["future"].set_result(row)

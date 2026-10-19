"""Per-request latency records. YOU implement to_record and drop_warmup.

GIVEN: the RequestRecord shape, so every result file has the same schema.

Definitions (these are the standard ones; vLLM's own benchmark uses the same):

  TTFT  time to first token: first content chunk time minus the moment the server received
        the request. Includes queueing and prefill.
  E2E   end-to-end latency: last content chunk time minus the same start.
  ITL   inter-token latency: the gaps between consecutive content chunks of ONE request,
        in seconds. A request that streams n chunks has n-1 gaps. (Speculation can deliver
        several tokens in one chunk; the gap is still chunk to chunk.)
  TPOT  time per output token: (E2E - TTFT) / (output_tokens - 1), the request's average
        decode speed. NaN-safe: a 1-token output has no TPOT, record 0.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import RawResult


@dataclass
class RequestRecord:
    rid: int
    start_s: float          # when the request was sent, sim seconds on the CLIENT clock (RawResult.send_sim)
    ttft_s: float
    e2e_s: float
    tpot_s: float
    itl_s: list = field(default_factory=list)   # per-gap list for this request
    prompt_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True


def to_record(raw: RawResult) -> RequestRecord:
    """YOU. Convert one RawResult into a RequestRecord.

    Contract (tests/test_bench.py, tests/test_experiments.py):
      * If raw.chunk_sim is not None (engine_sim): use the engine's clock. Start is
        raw.sim_arrival, chunk times are raw.chunk_sim. All durations are sim seconds.
      * Otherwise (a real server): start is raw.wall_start, chunk times are raw.chunk_wall,
        durations are wall seconds.
      * output_tokens is raw.completion_tokens when the server reported usage, else the
        number of chunks. Using the chunk count under speculation undercounts, which is why
        usage matters.
      * A failed request (raw.ok False) becomes a record with ok=False and zeros; it must
        not crash and it must not be silently dropped here (the analysis decides).
    """
    raise NotImplementedError("bench/records.py: to_record")


def drop_warmup(records: list, warmup_requests: int = 0, warmup_s: float = 0.0,
                cooldown_requests: int = 0) -> list:
    """YOU. Remove the transient at the start (and end) of a run so steady state is measured.

    Contract:
      * Order records by start_s first (do not trust the input order).
      * Drop the first `warmup_requests` records AND any record whose start_s is within
        `warmup_s` of the earliest start_s; drop the last `cooldown_requests`.
      * Never mutate the input list. If everything would be dropped, return [].
    Why it matters: the first requests of a closed loop all arrive together into an empty
    server and see a TTFT nobody will ever see again, and the last ones run alone.
    """
    raise NotImplementedError("bench/records.py: drop_warmup")

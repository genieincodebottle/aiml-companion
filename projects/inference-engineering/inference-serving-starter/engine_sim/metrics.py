"""Prometheus text exposition for engine_sim, using vLLM v1 metric names.

Written by hand (no prometheus_client) so the colon-style names come out exactly
as a real `vllm serve` exposes them. Counters carry the `_total` suffix the way a
scrape of a real server does.
"""
from __future__ import annotations

from bisect import bisect_left

# Bucket edges are in the style of vLLM's defaults (seconds); close, not identical.
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
                7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
ITL_BUCKETS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
               10.0, 20.0, 40.0, 80.0]
E2E_BUCKETS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
               120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]
QUEUE_BUCKETS = TTFT_BUCKETS


class Hist:
    def __init__(self, edges: list[float]):
        self.edges = edges
        self.counts = [0] * (len(edges) + 1)
        self.sum = 0.0
        self.count = 0

    def observe(self, v: float) -> None:
        self.counts[bisect_left(self.edges, v)] += 1
        self.sum += v
        self.count += 1


def _fmt(v: float) -> str:
    if isinstance(v, int):
        return str(v)
    return repr(round(v, 9))


class Metrics:
    def __init__(self) -> None:
        self.ttft = Hist(TTFT_BUCKETS)
        self.itl = Hist(ITL_BUCKETS)
        self.e2e = Hist(E2E_BUCKETS)
        self.queue = Hist(QUEUE_BUCKETS)
        self.preemptions = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.success = 0
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.spec_drafts = 0
        self.spec_draft_tokens = 0
        self.spec_accepted = 0

    def render(self, model_name: str, running: int, waiting: int, kv_usage: float,
               engine_id: str = "0") -> str:
        lab = f'model_name="{model_name}",engine="{engine_id}"'
        out: list[str] = []

        def gauge(name: str, help_: str, v) -> None:
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} gauge")
            out.append(f"{name}{{{lab}}} {_fmt(v)}")

        def counter(name: str, help_: str, v, extra: str = "") -> None:
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} counter")
            out.append(f"{name}{{{lab}{extra}}} {_fmt(v)}")

        def hist(name: str, help_: str, h: Hist) -> None:
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} histogram")
            cum = 0
            for edge, c in zip(h.edges, h.counts):
                cum += c
                out.append(f'{name}_bucket{{{lab},le="{edge}"}} {cum}')
            cum += h.counts[-1]
            out.append(f'{name}_bucket{{{lab},le="+Inf"}} {cum}')
            out.append(f"{name}_sum{{{lab}}} {_fmt(h.sum)}")
            out.append(f"{name}_count{{{lab}}} {h.count}")

        gauge("vllm:num_requests_running", "Number of requests in model execution batches.", running)
        gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", waiting)
        gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", kv_usage)
        counter("vllm:num_preemptions_total", "Cumulative number of preemptions from the engine.", self.preemptions)
        counter("vllm:prompt_tokens_total", "Number of prefill tokens processed.", self.prompt_tokens)
        counter("vllm:generation_tokens_total", "Number of generation tokens processed.", self.generation_tokens)
        counter("vllm:request_success_total", "Count of successfully processed requests.", self.success,
                ',finished_reason="length"')
        counter("vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens.",
                self.prefix_queries)
        counter("vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens.",
                self.prefix_hits)
        counter("vllm:spec_decode_num_drafts_total", "Number of spec decoding drafts.", self.spec_drafts)
        counter("vllm:spec_decode_num_draft_tokens_total", "Number of draft tokens.", self.spec_draft_tokens)
        counter("vllm:spec_decode_num_accepted_tokens_total", "Number of accepted tokens.", self.spec_accepted)
        hist("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", self.ttft)
        hist("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", self.itl)
        hist("vllm:e2e_request_latency_seconds", "Histogram of end to end request latency in seconds.", self.e2e)
        hist("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", self.queue)
        return "\n".join(out) + "\n"

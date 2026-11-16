"""OPTIONAL task: read a profiler trace of a toy decode step, on CPU. GIVEN.

    python profiling/profile_decode_step.py                 # batch 1, 8 and 64
    python profiling/profile_decode_step.py --batches 1,16 --layers 4

What this is. A stack of linear layers (hidden size H, L layers) stands in for the matmuls of one
decode step: each step reads every weight once and does 2 x batch x H x H arithmetic per layer. It runs
under torch.profiler, which records every operator. You get a table, and (with --trace-dir) a Chrome
trace to open at https://ui.perfetto.dev or chrome://tracing.

What this is NOT. A CPU profile teaches reading a trace; a GPU Nsight trace differs. On a GPU you see
CUDA kernels, launch gaps between them, and memory-throughput and tensor-core counters, and none of
those exist here. The toy model is float32 (no bf16 weights), it has no attention and no KV cache, and
a laptop CPU has a few tens of GB/s of memory bandwidth, not the hundreds of an L4. Use it to learn
the workflow (find the dominant operator, turn time into bytes per second, ask whether the step is
bound by memory or arithmetic), then repeat on the real GPU run with `vllm bench serve --profile`
or Nsight Systems, whichever your vLLM version documents.

Your task (about 45 minutes, not graded by `make check`):
  1. Run it. Which operator has the largest self CPU time, and what share of the total is it?
  2. For batch 1, compute the weight bytes read per step divided by the step time: GB/s. Is it
     roughly constant as the batch grows, or does it fall? Why?
  3. At which batch does the step time start to grow clearly? That batch is where this CPU stops
     being memory-bound. Write one paragraph for your report: the same experiment on your GPU
     would put the knee at a different batch, and the ridge point (peak FLOPs / peak bandwidth,
     see engine_sim/config.py) predicts roughly where.

For scale only, one run on a laptop CPU with 8 threads (2026-11-16, your numbers will differ):
batch 1 took 6.8 ms (19.8 GB/s of weights read), batch 8 took 13.3 ms and batch 64 took 28.4 ms, and
aten::mm held 83 to 94 percent of the self CPU time. On that machine the step was already
compute-limited by batch 8, which is not what a GPU does: that is the point of question 3.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def build(hidden: int, layers: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(*[torch.nn.Linear(hidden, hidden, bias=False) for _ in range(layers)]).eval()


def step_time(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> float:
    with torch.no_grad():
        for _ in range(3):          # warm-up: allocator, thread pool
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        return (time.perf_counter() - t0) / repeats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batches", default="1,8,64")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--trace-dir", default=None, help="write one Chrome trace per batch here")
    a = ap.parse_args()

    model = build(a.hidden, a.layers)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"toy decode step: {a.layers} layers of {a.hidden} x {a.hidden}, float32, "
          f"{weight_bytes / 1e6:.0f} MB of weights, {torch.get_num_threads()} CPU threads")
    print(f"{'batch':>6} {'step ms':>9} {'GB/s read':>10} {'GFLOP/s':>9}")
    for b in [int(v) for v in a.batches.split(",")]:
        x = torch.randn(b, a.hidden)
        t = step_time(model, x, a.repeats)
        flops = 2.0 * b * a.hidden * a.hidden * a.layers
        print(f"{b:>6} {t * 1e3:9.2f} {weight_bytes / t / 1e9:10.1f} {flops / t / 1e9:9.1f}")

    for b in [int(v) for v in a.batches.split(",")]:
        x = torch.randn(b, a.hidden)
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function(f"decode_step_batch_{b}"):
                for _ in range(5):
                    model(x)
        print(f"\n--- operator table, batch {b} (5 steps) ---")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=6))
        if a.trace_dir:
            Path(a.trace_dir).mkdir(parents=True, exist_ok=True)
            out = Path(a.trace_dir) / f"cpu_profile_batch{b}.json"
            prof.export_chrome_trace(str(out))
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

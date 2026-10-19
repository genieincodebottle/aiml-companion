"""Run the five experiments the report needs, against engine_sim. GIVEN orchestration.

    python scripts/run_experiments.py            # run whatever is missing
    python scripts/run_experiments.py --force    # redo everything
    python scripts/run_experiments.py --only quant_bf16

For each experiment it starts `python -m engine_sim ...` on a free port, runs
`python -m bench.cli sweep ...` (your load generator), and stops the server. It fails until
bench/ is implemented. Results land in results/*.json, which the tests read.

EXPERIMENT 1, quantisation (NVIDIA L4, Llama-3.1-8B, bf16 vs fp8 vs int4, speculation off).
EXPERIMENT 2, speculation (NVIDIA A10, Llama-3.1-8B bf16, ngram off vs on, k=4, acceptance 0.5,
short-context chat workload). The acceptance rate is an assumption of the model (see README),
so the speculation result is about the SHAPE (a win at low concurrency, a loss at high), not
about the size of the win.
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

QUANT_LEVELS = "1,2,4,8,16,32,64,128"
SPEC_LEVELS = "1,4,16,64,128,256"
CHAT = str(ROOT / "bench" / "data" / "length_pairs_chat.jsonl")

# name: (engine_sim args, concurrency levels, time_scale, dataset or None for the default)
# The fast configs (A10 speculation) get a lower time_scale: at a high speed-up the HTTP layer,
# not the engine, becomes the bottleneck and the closed loop is slowed by wall-clock overhead.
EXPERIMENTS = {
    "quant_bf16": (["--gpu", "l4", "--quant", "bf16"], QUANT_LEVELS, 10.0, None),
    "quant_fp8": (["--gpu", "l4", "--quant", "fp8"], QUANT_LEVELS, 10.0, None),
    "quant_int4": (["--gpu", "l4", "--quant", "int4"], QUANT_LEVELS, 10.0, None),
    "spec_off": (["--gpu", "a10", "--quant", "bf16", "--spec", "off"], SPEC_LEVELS, 5.0, CHAT),
    "spec_ngram": (["--gpu", "a10", "--quant", "bf16", "--spec", "ngram", "--spec-k", "4",
                    "--spec-accept", "0.5"], SPEC_LEVELS, 5.0, CHAT),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_one(name: str, server_args: list, levels: str, time_scale: float, dataset, seed: int) -> int:
    port = free_port()
    cmd = [sys.executable, "-m", "engine_sim", "--port", str(port), "--time-scale", str(time_scale),
           "--seed", str(seed)] + server_args
    print(f"[{name}] starting: {' '.join(cmd[2:])}", flush=True)
    server = subprocess.Popen(cmd, cwd=ROOT)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            print(f"[{name}] server did not start", file=sys.stderr)
            return 1
        out = ROOT / "results" / f"{name}.json"
        cli = [sys.executable, "-m", "bench.cli", "sweep", "--base-url", base, "--name", name,
               "--levels", levels, "--out", str(out)]
        if dataset:
            cli += ["--dataset", dataset]
        return subprocess.call(cli, cwd=ROOT)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--time-scale", type=float, default=None, help="override the per-experiment speed-up")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rc = 0
    for name, (args, levels, scale, dataset) in EXPERIMENTS.items():
        if a.only and a.only != name:
            continue
        if (ROOT / "results" / f"{name}.json").exists() and not a.force:
            print(f"[{name}] exists, skipping (use --force)")
            continue
        rc |= run_one(name, args, levels, a.time_scale or scale, dataset, a.seed)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

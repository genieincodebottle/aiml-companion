"""python -m engine_sim --model 8b --gpu l4 --quant bf16 --spec off --time-scale 10 --port 8100"""
from __future__ import annotations

import argparse
import sys

import uvicorn

from .config import GPUS, MODELS, QUANT_BYTES, EngineConfig
from .server import create_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="engine_sim", description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="8b")
    p.add_argument("--gpu", choices=sorted(GPUS), default="l4")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--quant", choices=sorted(QUANT_BYTES), default="bf16")
    p.add_argument("--spec", choices=["off", "ngram"], default="off")
    p.add_argument("--spec-k", type=int, default=4)
    p.add_argument("--spec-accept", type=float, default=0.6, help="per-draft-token acceptance probability")
    p.add_argument("--time-scale", type=float, default=10.0, help="sim seconds per wall second")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    a = p.parse_args(argv)
    cfg = EngineConfig(
        model=a.model, gpu=a.gpu, num_gpus=a.num_gpus, quant=a.quant, spec=a.spec, spec_k=a.spec_k,
        spec_accept=a.spec_accept, time_scale=a.time_scale, seed=a.seed, max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens, enable_prefix_caching=not a.no_prefix_caching,
    )
    try:
        cfg.validate()
    except ValueError as e:
        print(f"engine_sim: {e}", file=sys.stderr)
        return 2
    print(f"engine_sim: {cfg.describe()}", flush=True)
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

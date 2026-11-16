"""OPTIONAL extension, the real-GPU step: record one config's outputs on the fixed prompt set. GIVEN.

    # terminal 1 (GPU machine): start ONE config, e.g. bash gpu/serve_bf16.sh
    # terminal 2:
    python quality/run_prompts.py --base-url http://127.0.0.1:8000 --name bf16
    # stop the server, start the next config (serve_ngram.sh, serve_fp8.sh, serve_awq.sh), then:
    python quality/run_prompts.py --base-url http://127.0.0.1:8000 --name ngram
    python quality/check_equivalence.py --baseline quality/outputs/bf16.jsonl \
        --candidate quality/outputs/ngram.jsonl --min-exact 0.9 --min-prefix 0.95

It sends each prompt in quality/prompts.jsonl to /v1/completions, one at a time, at temperature 0
with max_tokens 128, and writes quality/outputs/<name>.jsonl. One request at a time keeps the batch
size at 1, so a batching effect cannot change the answer between configs: that makes the comparison
about the config and not about load.

engine_sim has no text. Its server returns placeholder words, so it cannot show a quality change,
and this script refuses to run against it rather than record meaningless outputs. Use the same
MODEL for every config you compare, and keep the same vLLM version.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--name", required=True, help="output file stem, e.g. bf16, fp8, ngram")
    ap.add_argument("--model", default=None, help="default: the first model the server lists")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    a = ap.parse_args(argv)
    base = a.base_url.rstrip("/")
    with httpx.Client(timeout=300) as client:
        if client.get(base + "/sim/info").status_code == 200:
            print("error: this is engine_sim, which has no text. Point --base-url at a real vllm serve.",
                  file=sys.stderr)
            return 2
        model = a.model or client.get(base + "/v1/models").json()["data"][0]["id"]
        rows = [json.loads(line) for line in Path(a.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
        out = []
        for i, row in enumerate(rows, 1):
            r = client.post(base + "/v1/completions", json={
                "model": model, "prompt": row["prompt"], "max_tokens": a.max_tokens, "temperature": 0})
            r.raise_for_status()
            choice = r.json()["choices"][0]
            out.append({"id": row["id"], "kind": row["kind"], "text": choice["text"],
                        "finish_reason": choice.get("finish_reason")})
            print(f"  {i}/{len(rows)} {row['id']}", file=sys.stderr, flush=True)
    path = Path(a.out_dir) / f"{a.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f"wrote {path} ({len(out)} outputs from {model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

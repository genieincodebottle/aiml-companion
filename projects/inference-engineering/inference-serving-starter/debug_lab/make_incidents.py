"""Regenerate the three recorded incident bundles. GIVEN. Deterministic.

    python debug_lab/make_incidents.py            # rewrite debug_lab/incidents/
    python debug_lab/make_incidents.py --out DIR  # write somewhere else (the test does this)

Each bundle is what an on-call engineer would have after the page fires: Prometheus text
scrapes every 30 s of engine time, plus the engine's own log lines. The faults are
injected by running engine_sim with a hostile workload or configuration. The README in
each folder is the ticket the customer filed. It names the symptom only.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine_sim.config import EngineConfig  # noqa: E402
from engine_sim.offline import simulate  # noqa: E402
from engine_sim.workload import make_requests, poisson_arrivals  # noqa: E402

DURATION_S = 480.0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _dump(out: Path, res: dict, sub: str = "") -> None:
    base = out / sub if sub else out
    for i, (t, text) in enumerate(res["scrapes"]):
        _write(base / "scrapes" / f"scrape_{i:02d}_t{int(t):04d}s.prom", text)


# ---------------------------------------------------------------------------------------
def incident_a(out: Path) -> None:
    """Customer: 'Chat latency fell off a cliff after the marketing push at about minute 2.'
    Fault: arrival rate triples on an L4 whose KV cache holds about 30k tokens."""
    cfg = EngineConfig(model="8b", gpu="l4", quant="bf16", seed=11)
    rng = random.Random(1101)
    arr = poisson_arrivals([(0, 120, 0.25), (120, DURATION_S, 0.9)], rng)
    reqs = make_requests(arr, rng, prompt_median=900, prompt_sigma=0.5, out_median=350, out_sigma=0.4,
                         prompt_range=(100, 3000), out_range=(40, 900))
    res = simulate(cfg, reqs, DURATION_S, label="a")
    _dump(out / "a_kv_saturation", res)
    _write(out / "a_kv_saturation" / "logs.txt", "\n".join(res["logs"]) + "\n")
    _write(out / "a_kv_saturation" / "TICKET.md", TICKET_A)


# ---------------------------------------------------------------------------------------
def incident_b(out: Path) -> None:
    """Customer: 'Two replicas behind our router, but p99 TTFT is awful and one GPU looks idle.'
    Fault: the router hashes the system-prompt id to pick a replica, and one system prompt
    carries 75 percent of the traffic."""
    cfg = EngineConfig(model="8b", gpu="a10", quant="bf16", seed=22)
    rng = random.Random(2202)
    arr = poisson_arrivals([(0, DURATION_S, 6.5)], rng)
    groups = [(0.75, 0, 1500)] + [(0.25 / 6, g, 400) for g in range(1, 7)]
    reqs = make_requests(arr, rng, prompt_median=150, prompt_sigma=0.5, out_median=200, out_sigma=0.4,
                         prompt_range=(20, 800), out_range=(30, 600), prefix_groups=groups)
    per_replica: list = [[], []]
    for r in reqs:
        gid = int(r.tag[1:])
        per_replica[zlib.crc32(f"prefix-{gid}".encode()) % 2].append(r)
    for k in (0, 1):
        res = simulate(cfg.with_(seed=22 + k), per_replica[k], DURATION_S, engine_id=str(k), label=f"replica{k}")
        _dump(out / "b_hot_prefix_imbalance", res, f"replica{k}")
        _write(out / "b_hot_prefix_imbalance" / f"logs_replica{k}.txt", "\n".join(res["logs"]) + "\n")
    _write(out / "b_hot_prefix_imbalance" / "TICKET.md", TICKET_B)


# ---------------------------------------------------------------------------------------
def incident_c(out: Path) -> None:
    """Customer: 'We turned on n-gram speculation last week. Mornings are great, afternoons are worse
    than before we enabled it.'  Fault: speculation on, load ramps to a batch far above the ridge."""
    cfg = EngineConfig(model="8b", gpu="a10", quant="bf16", spec="ngram", spec_k=4, spec_accept=0.5,
                       seed=33, max_num_seqs=128)
    rng = random.Random(3303)
    arr = poisson_arrivals([(0, 120, 1.0), (120, DURATION_S, 14.0)], rng)
    reqs = make_requests(arr, rng, prompt_median=80, prompt_sigma=0.4, out_median=150, out_sigma=0.3,
                         prompt_range=(24, 400), out_range=(40, 400))
    res = simulate(cfg, reqs, DURATION_S, label="c")
    _dump(out / "c_speculation_collapse", res)
    _write(out / "c_speculation_collapse" / "logs.txt", "\n".join(res["logs"]) + "\n")
    _write(out / "c_speculation_collapse" / "TICKET.md", TICKET_C)


TICKET_A = """# Ticket A: chat latency cliff

Model: Llama-3.1-8B bf16 on one NVIDIA L4 (engine_sim). Traffic was light for two minutes,
then a campaign went out. Users now report chat replies that start after many seconds and
then stall mid-answer. GPU utilisation looks fine, nothing crashed, error rate is zero.

Scrapes every 30 s of engine time in scrapes/, engine log lines in logs.txt.
Write your diagnosis in debug_lab/diagnoses/a_kv_saturation.md.
"""

TICKET_B = """# Ticket B: two replicas, one idle

Two identical replicas (Llama-3.1-8B bf16 on one NVIDIA A10 each) behind a router that
sends each request to a replica chosen from its prompt prefix, so prefix caching works.
Users report slow first tokens. The dashboard shows fleet-average GPU utilisation of 60 percent.

Scrapes for each replica in replica0/scrapes and replica1/scrapes, logs in logs_replica*.txt.
Write your diagnosis in debug_lab/diagnoses/b_hot_prefix_imbalance.md.
"""

TICKET_C = """# Ticket C: speculation made the afternoon worse

Llama-3.1-8B bf16 on one NVIDIA A10 with n-gram speculative decoding (4 draft tokens).
Quiet periods are faster than before speculation. At busy times, tokens per second and
inter-token latency are worse than they were with speculation off. The team suspects the
draft model quality (n-gram acceptance) dropped.

Scrapes every 30 s of engine time in scrapes/, engine log lines in logs.txt.
Write your diagnosis in debug_lab/diagnoses/c_speculation_collapse.md.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "debug_lab" / "incidents"))
    a = ap.parse_args()
    out = Path(a.out)
    for name in ("a_kv_saturation", "b_hot_prefix_imbalance", "c_speculation_collapse"):
        if (out / name).exists():
            shutil.rmtree(out / name)
    incident_a(out)
    incident_b(out)
    incident_c(out)
    print(f"wrote incident bundles to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

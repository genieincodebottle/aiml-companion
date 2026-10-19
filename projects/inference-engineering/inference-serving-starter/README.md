# Inference Serving Starter

> **Used in the Inference Engineering capstone on [AI-ML Companion](https://aimlcompanion.ai/): "Project: Serve, Benchmark and Cost an Open Model".**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status: Starter](https://img.shields.io/badge/Status-Starter_Scaffold-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**This repo is deliberately incomplete. Five of the six gates fail when you clone it. That is the point.**

You walk away with portfolio evidence for an inference-engineer job:

> "I served an open model, benchmarked it properly, compared quantisation and speculative decoding, diagnosed three incidents from metrics, and produced a cost-per-million-tokens figure and a fleet-size recommendation."

The hard-to-fake parts are given to you: a serving engine to benchmark, a dataset, the incident recordings, and the definition of done written as failing tests. You write the load generator, the analysis and the report.

```
engine_sim/   GIVEN  A deterministic model of a vLLM-style server (OpenAI API + /metrics).
bench/        YOU    Open-loop and closed-loop load generators, per-request records.
analysis/     YOU    Percentiles, goodput, knee plot, cost per 1M tokens, fleet size.
debug_lab/    GIVEN data, YOU diagnose. Three recorded incidents.
gpu/          GIVEN  The same harness against a real `vllm serve` (where numbers become real).
k8s/          GIVEN  Stretch: autoscale engine_sim on queue depth with kind. Not gated.
report/       YOU    One-page report template with "what a weak version looks like" notes.
tests/        GIVEN  The rubric, as executable spec.
```

## Honest scope

`engine_sim` is a **model, not a GPU**. Its latency model is documented and physics-based
(memory-bound decode, compute-bound prefill, finite KV cache, preemption, prefix caching), with
simplifications listed in `engine_sim/core.py`. It reproduces the *shape* of real behaviour and
the *signatures* in the metrics. It cannot tell you what an L4 will do. The GPU path in `gpu/` is
where numbers become real. Your report must say which of its numbers are model output.

Hardware figures are vendor spec-sheet values with the source URL in `engine_sim/config.py`.
Everything else (bandwidth efficiency, MFU, n-gram acceptance rate) is an **assumption**, labelled
as such. There is no GPU price list in this repo: you bring the price and its source.

## Quick start (3 steps, CPU only, no API key)

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/inference-engineering/inference-serving-starter
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run.py check          # make check: six gates, one green on clone
python run.py serve          # in a second shell: engine_sim on :8100  (curl localhost:8100/metrics)
```

`make` is optional: `python run.py <task>` does the same. Run `python run.py` for the task list.

## The six gates

| # | Gate | You build | Done when |
|---|---|---|---|
| 1 | Given engine | nothing | passes on clone: decode throughput rises with batch until KV is full, prefix hits cut TTFT, int4 beats bf16 at batch 1, speculation flips sign, incident bundles regenerate byte for byte |
| 2 | Harness | `bench/dataset.py`, `loadgen.py`, `records.py` | Poisson arrivals are really Poisson, the open loop keeps its schedule while requests are slow, the closed loop holds exactly N in flight, TTFT/ITL/E2E match the server's own histograms |
| 3 | Analysis | `analysis/tables.py`, `goodput.py`, `knee.py`, `cost.py`, `fleet.py` | pooled ITL percentiles, goodput under a TTFT+TPOT SLO, a knee, `$ per 1M tokens`, Little's-law fleet size |
| 4 | Experiments | run `scripts/run_experiments.py` | quantisation and speculation sweeps are physically sensible, a knee plot exists |
| 5 | Debug lab | three files in `debug_lab/diagnoses/` | each names the mechanism and cites real numbers from the scrapes |
| 6 | Report | `report/REPORT.md`, `results/decision.json` | finished one-pager quoting your own cost and replica count |

Gate 4 asserts physics (for example "int4 beats bf16 at concurrency 1", "speculation wins at low
concurrency and loses at high"). A red line there is information about **your** code.

## 7-day plan

| Day | Work | Gate |
|---|---|---|
| 1 | Read `engine_sim/core.py` docstring and `bench/*.py` contracts. Implement `sample_workload`, `poisson_arrival_times`, `run_closed_loop`. Run one request by hand with `curl` | start 2 |
| 2 | `run_open_loop`, `to_record`, `drop_warmup`, `concurrency_sweep`. Make `test_harness_agrees_with_the_servers_histograms` pass: if your TTFT disagrees with the server's, you have a bug | 2 |
| 3 | `analysis/`: percentile, pooled ITL, goodput, knee. Then `python scripts/run_experiments.py` and look at the tables | 3, 4 |
| 4 | Read the quantisation and speculation results. For each surprise, find the mechanism in `core.py`. Write sections 4 and 5 of the report | 4 |
| 5 | Debug lab: diagnose A, B, C from the scrapes. Plot the metrics over time before you guess | 5 |
| 6 | Cost and fleet: `python -m analysis.decide`. Add your price and its source. GPU path if you have access (Colab T4 is free) | 6 |
| 7 | Finish the report against the "WEAK:" notes. Self-score the rubric below and spend the last two hours on your weakest row | 6 |

## Rubric (what a reviewer scores)

| Criterion | Weight | What earns it |
|---|---|---|
| Benchmark method | 25 | open vs closed loop chosen on purpose, warm-up excluded, pooled ITL, harness cross-checked against server metrics |
| Analysis correctness | 20 | percentiles over the right population, goodput under an explicit SLO, a defensible knee |
| Comparative experiments | 20 | quantisation and speculation explained by mechanism (weight bytes, KV room, ridge point), not just reported |
| Debugging from data | 15 | the right mechanism, numbers cited from scrapes, a fix and the metric that would confirm it |
| Cost and fleet decision | 10 | cost from throughput at the SLO-safe operating point, a sourced price, Little's law with headroom |
| Honesty | 10 | model vs measurement separated, threats to validity that name a direction |

## What you do with the result

Put the repo (with your `bench/`, `analysis/`, results, diagnoses and `report/REPORT.md`) on your
own GitHub and link the report at the top of the README. In an interview, be ready for: "why is
your closed-loop p99 lower than your open-loop p99", "what changes on a real GPU", and "what
would you measure first on the real card".

## Safety

No secrets are needed or stored here. The GPU path needs a Hugging Face token for gated models:
keep it in an environment variable, never in a file. Stop rented GPUs when you are done.

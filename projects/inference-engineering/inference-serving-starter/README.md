# Inference Serving Starter

> **Used in the Inference Engineering capstone on [AI-ML Companion](https://aimlcompanion.ai/): "Project: Serve, Benchmark and Cost an Open Model".**

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Status: Starter](https://img.shields.io/badge/Status-Starter_Scaffold-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**This repo is deliberately incomplete. Five of the six gates fail when you clone it. That is the point.**

You walk away with portfolio evidence for an inference-engineer job:

> "I served an open model, benchmarked it properly with an open loop and a closed loop, compared quantisation and speculative decoding, diagnosed three incidents from metrics, and produced a GPU-hours-per-million-tokens figure under two SLOs, a self-host-versus-API break-even and a fleet-size recommendation."

The hard-to-fake parts are given to you: a serving engine to benchmark, a dataset, the incident recordings, and the definition of done written as failing tests. You write the load generator, the analysis and the report.

```
engine_sim/   GIVEN  A deterministic model of a vLLM-style server (OpenAI API + /metrics).
bench/        YOU    Open-loop and closed-loop load generators, per-request records.
analysis/     YOU    Percentiles, goodput, knee plot, GPU-hours per 1M tokens, fleet size.
debug_lab/    GIVEN data, YOU diagnose. Three recorded incidents, and an optional fourth with a config diff.
gpu/          GIVEN  The same harness against a real `vllm serve` (where numbers become real), incl. a TP=2 script.
k8s/          GIVEN  Stretch: autoscale engine_sim on queue depth; a gang-scheduling lab. Not gated.
profiling/    GIVEN  Optional: read a torch.profiler trace of a toy decode step on CPU.
quality/      GIVEN  Optional: a quality-equivalence gate (needs the real-GPU session).
report/       YOU    Short report template with "what a weak version looks like" notes.
tests/        GIVEN  The rubric, as executable spec.
```

## Honest scope

`engine_sim` is a **model, not a GPU**. Its latency model is documented and physics-based
(memory-bound decode, compute-bound prefill, finite KV cache, preemption, prefix caching), with
simplifications listed in `engine_sim/core.py`. It reproduces the *shape* of real behaviour and
the *signatures* in the metrics. It cannot tell you what an L4 will do. The GPU path in `gpu/` is
where numbers become real, and one short run there is **required**: Gate 6 checks for a real-GPU
sweep in `results/` and a "Measured on a real GPU" section in the report. Budget 2 to 3 hours: a free
Colab T4 when one is available, or a rented L4 for about 2 to 3 GPU-hours (see `gpu/README.md`). Your
report must say which of its numbers are model output and which were measured.

Hardware figures are vendor spec-sheet values with the source URL in `engine_sim/config.py`.
Everything else (bandwidth efficiency, MFU, n-gram acceptance rate) is an **assumption**, labelled
as such. This repo works in GPU time (GPU-hours and GPU-seconds), not money: it holds no price list.

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
| 3 | Analysis | `analysis/tables.py`, `goodput.py`, `knee.py`, `cost.py`, `fleet.py` | pooled ITL percentiles, goodput under a TTFT+TPOT SLO, a knee, `GPU-hours per 1M tokens`, Little's-law fleet size |
| 4 | Experiments | run `scripts/run_experiments.py` (about 8 to 10 minutes) | quantisation and speculation sweeps are physically sensible, an open-loop run shows the queue a closed loop hides, a knee plot exists |
| 5 | Debug lab | three files in `debug_lab/diagnoses/` | each names the mechanism, cites real numbers from two or more scrapes (a counter from two scrapes of one folder), and gives a FIX that names the metric that would confirm it |
| 6 | Report | `report/REPORT.md`, `results/decision.json`, one real-GPU sweep in `results/` | finished short report quoting your own GPU-hours per 1M tokens and replica count under two SLOs, the open-loop p99, the self-host-versus-API break-even, and the real-GPU run set beside the model's numbers |

Gate 4 asserts physics (for example "int4 beats bf16 at concurrency 1", "speculation wins at low
concurrency and loses at high"). A red line there is information about **your** code.

## 7-day plan

About 14 hours of laptop work over seven days (2 h a day, a little less on day 6, a little more on day 7), plus
the real-GPU session on day 6.

| Day | Hours | Work | Gate |
|---|---|---|---|
| 1 | 2 | Read `engine_sim/core.py` docstring and `bench/*.py` contracts. Implement `sample_workload`, `poisson_arrival_times`, `run_closed_loop`. Run one request by hand with `curl` | start 2 |
| 2 | 2 | `run_open_loop`, `to_record`, `drop_warmup`, `concurrency_sweep`. Make `test_harness_agrees_with_the_servers_histograms` pass: if your TTFT disagrees with the server's, you have a bug | 2 |
| 3 | 2 | `analysis/`: percentile, pooled ITL, goodput, knee. Then `python scripts/run_experiments.py` (about 8 to 10 minutes) and look at the tables | 3, 4 |
| 4 | 2 | Read the quantisation, speculation and open-loop results. For each surprise, find the mechanism in `core.py`. Write sections 3 to 5 of the report, including the open-loop paragraph | 4 |
| 5 | 2 | Debug lab: diagnose A, B, C from the scrapes. Plot the metrics over time before you guess. Read two scrapes of each counter. Optional: incident D | 5 |
| 6 | 1.5 + 2 to 3 GPU | GPU usage and fleet: `python -m analysis.decide`, with the API rate expressed as GPU-hours of your GPU per 1M tokens and how you derived it. Then the real-GPU session (`gpu/README.md`): a bf16 sweep (fp16 on a T4, which has no bf16), the `vllm bench serve` cross-check, and section 8 of the report | 6 |
| 7 | 2.5 | Finish the report against the "WEAK:" notes. Self-score the rubric below and spend the last two hours on your weakest row | 6 |

## Rubric (what a reviewer scores)

| Criterion | Weight | What earns it |
|---|---|---|
| Benchmark method | 20 | open vs closed loop chosen on purpose and shown on your own data (the open-loop run), warm-up excluded, pooled ITL, harness cross-checked against server metrics |
| Analysis correctness | 20 | percentiles over the right population, goodput under an explicit SLO, a defensible knee |
| Comparative experiments | 15 | quantisation and speculation explained by mechanism (weight bytes, KV room, ridge point), not just reported |
| Debugging from data | 15 | the right mechanism, numbers cited from scrapes, a fix and the metric that would confirm it |
| Real-GPU evidence | 10 | one real `vllm serve` run through your harness: GPU, vLLM version and flags stated, cross-checked against `vllm bench serve`, and each moved number explained |
| GPU usage and fleet decision | 10 | GPU-hours per 1M tokens from throughput at the SLO-safe operating point under two SLOs (plus the effective figure at peak utilisation), input-token usage as GPU-seconds per request, a self-host-versus-API break-even utilisation with the API rate and its derivation stated, Little's law with headroom |
| Honesty | 10 | model vs measurement separated, threats to validity that name a direction |

## Optional extensions (not gated)

| Extension | Where | What it teaches | Status when written |
|---|---|---|---|
| Incident D: a before/after regression with a config diff | `debug_lab/incidents/d_config_regression/` | rule out a suspect with a number; read two folders | checked by `check_diagnosis.py` if you write the file |
| CPU profile of a toy decode step | `profiling/` (needs `pip install torch`, CPU build is enough) | reading an operator table and a trace; a CPU profile teaches the workflow, a GPU Nsight trace differs | run on a laptop CPU |
| Tensor parallel across 2 GPUs | `gpu/serve_tp2.sh`, `gpu/README.md` | whether a second GPU pays for itself | **not run**, no two-GPU machine |
| Quality-equivalence gate | `quality/` | agreement of fp8 or speculation outputs with bf16 on 32 fixed prompts | comparator tested on synthetic outputs; recording step **not run** (needs a real server) |
| Gang-scheduling deadlock on kind | `k8s/gang_lab/` | why multi-GPU jobs need gang admission | **not run**: manifest and a scheduler model checked offline, no kind on the machine |

## What you do with the result

Put the repo (with your `bench/`, `analysis/`, results, diagnoses and `report/REPORT.md`) on your
own GitHub and link the report at the top of the README. In an interview, be ready for: "why is
your closed-loop p99 lower than your open-loop p99", "what changes on a real GPU", and "what
would you measure first on the real card".

## Safety

No secrets are needed or stored here. The GPU path needs a Hugging Face token for gated models:
keep it in an environment variable, never in a file. Stop rented GPUs when you are done.

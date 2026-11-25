# GPU path: where the numbers become real

`engine_sim` is a model. It teaches the shape of the curves and the metric signatures, and it
cannot tell you what an L4 will really do. This folder runs the **same** harness (`bench/`,
`analysis/`) against a real `vllm serve`, so the numbers in your report can be real.

Gate 6 needs one real run: a sweep from this folder saved in `results/` (for example
`results/gpu_bf16.json`) and the report's "Measured on a real GPU" section. Two to three hours on
the smallest machine below is enough.

**Budget, stated up front:** a free Colab T4 is available when Colab has one, but Colab does not
guarantee GPU access. The session is about 2 to 3 GPU-hours on a rented L4 (24 GB). Stop the instance
when you finish.

## What was checked, and when

The flags below were read from the vLLM docs on 2026-10-19. The pinned version is **v0.33.0**
(released 2026-11-02). It is pinned so a fresh clone stays reproducible; re-check the flags if you
bump it.

On 2026-11-14 the pin was run on a T4 (16 GB) with `Qwen/Qwen2.5-1.5B-Instruct` and the T4 settings
from `colab_t4.md`: `install.sh` (including `vllm download-kernels`), then `serve_bf16.sh` with
`DTYPE=half`, `serve_ngram.sh`, `serve_fp8.sh` and `serve_awq.sh` (with
`MODEL_AWQ=Qwen/Qwen2.5-1.5B-Instruct-AWQ`). Every server started, answered completions and chat,
and exposed the metric names this lab reads, including the `vllm:spec_decode_*` counters under
n-gram speculation. `bench_crosscheck.sh` completed. No throughput numbers from that run are quoted
here: a T4 is not the machine your report is about.

`serve_tp2.sh` (tensor parallel across 2 GPUs) was added on 2026-11-16 and was **not run**: that session had no
two-GPU machine. Its one new flag, `--tensor-parallel-size`, comes from
https://docs.vllm.ai/en/latest/serving/parallelism_scaling/ (read 2026-11-16).

The scripts here are **not executed by this repository's tests** (there is no GPU in CI). Treat
the first run as part of the exercise: if a flag is rejected, `vllm serve --help=all` shows what
your version accepts.

| Fact | Source (read 2026-10-19) |
|---|---|
| `--speculative-config '{"method":"ngram","num_speculative_tokens":N,"prompt_lookup_max":M}'` | https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ |
| `--quantization fp8_per_tensor` is the documented online FP8 mode; FP8 compute needs compute capability 8.9 or higher (Ada, Hopper, Blackwell). Turing and Ampere get weight-only FP8 (W8A16) | https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/ |
| `vllm bench serve` flags: `--backend --base-url --endpoint --dataset-name --random-input-len --random-output-len --num-prompts --request-rate --max-concurrency --goodput --ignore-eos --save-result --result-dir --percentile-metrics --metric-percentiles` | https://docs.vllm.ai/en/latest/cli/bench/serve/ |
| Install: `uv pip install vllm --torch-backend=auto`, then `vllm download-kernels`; compute capability 7.5 or higher (T4, L4, A10, H100 all qualify) | https://docs.vllm.ai/en/latest/getting_started/installation/gpu/ |
| Metric names (`vllm:kv_cache_usage_perc`, `vllm:inter_token_latency_seconds`, `vllm:time_to_first_token_seconds`, prefix cache counters) | https://docs.vllm.ai/en/latest/design/metrics/ |
| `--gpu-memory-utilization` (0.92 in older releases; recent CUDA builds leave it unset, which means 1.0), plus `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-prefix-caching` | https://docs.vllm.ai/en/latest/cli/serve/ |

The docs for the older FP8 spelling `--quantization fp8` differ by version. If your version
rejects `fp8_per_tensor`, run `vllm serve --help=all` and use the spelling it lists. Do not use a
flag your version's docs do not list. Speculation is configured only through `--speculative-config`;
the older separate speculative flags were removed.

## Pick a machine

| Option | Model | Works for |
|---|---|---|
| Free Colab T4 (16 GB) | a 1 to 3B model, e.g. `Qwen/Qwen2.5-1.5B-Instruct` | the whole loop, small numbers. See `colab_t4.md` |
| Rented L4 (24 GB) | `meta-llama/Llama-3.1-8B-Instruct` (gated: accept the licence, set `HF_TOKEN`) or `Qwen/Qwen3-8B` | the quantisation experiment (L4 has FP8 tensor cores) |
| Rented A10 or A10G (24 GB) | same 8B models | bf16 and AWQ. No FP8 tensor cores |
| Rented H100 (80 GB) | same | the speculation experiment at high concurrency |
| Two GPUs in one machine (2 x L4, 2 x A10, 2 x H100) | same | the optional TP=2 comparison (`serve_tp2.sh`), see below |

### Optional: tensor parallel across 2 GPUs

An 8B model fits on one 24 GB card, so TP=2 is not needed for it. It is worth one extra sweep because
the question "does a second GPU double my throughput?" has a measurable answer, and the answer depends on the
link. Run `nvidia-smi topo -m` first: `NV#` means NVLink, `PHB`, `PIX` or `SYS` means PCIe. Then run
`bash gpu/serve_bf16.sh` and sweep it, stop it, run `bash gpu/serve_tp2.sh` on the same model and sweep it
again with the same levels. Report tokens per second per GPU for both, and say what the all-reduce did.
The vLLM docs recommend tensor parallelism for a model too big for one GPU and pipeline parallelism for
cards without NVLink such as the L40S, so expect a PCIe pair to scale poorly.
Two GPUs double the GPU-hours you burn per hour: your GPU-hours per million tokens only improves if throughput more than doubles.

Usage, not money: this repo counts GPU-hours and GPU-seconds. Write the GPU, the vLLM version and the
date of your run in the report; the repo holds no price table, on purpose.

## The loop (same harness, real server)

```bash
# 1. on the GPU machine
export HF_TOKEN=...            # never commit it; no token is in this repo
export MODEL=meta-llama/Llama-3.1-8B-Instruct
bash gpu/install.sh            # pins vllm==0.33.0
bash gpu/serve_bf16.sh         # listens on :8000; wait for "Application startup complete"

# 2. same machine (or point --base-url at it), in a second shell
python -m bench.cli sweep --base-url http://127.0.0.1:8000 --name gpu_bf16 \
    --levels 1,2,4,8,16,32 --out results/gpu_bf16.json
python -m analysis.summarize results/gpu_bf16.json --plot results/knee_gpu_bf16.png

# 3. cross-check your harness against vLLM's own benchmark (should agree within about 10 percent)
bash gpu/bench_crosscheck.sh

# 4. stop the server, change the flags (serve_fp8.sh, serve_awq.sh, serve_ngram.sh), repeat
# 5. STOP THE INSTANCE. An idle H100 bills by the hour.
```

`bench.cli` detects a real server automatically: `/sim/info` returns 404, so it uses wall-clock
time, a `time_scale` of 1, and the usage block for token counts. Your `to_record` handles both
because the contract in `bench/records.py` says it must.

## What will differ from engine_sim, and why you should expect it

| engine_sim | Real vLLM |
|---|---|
| One engine step costs `max(memory, compute) + overhead`, exactly | Real kernels, CUDA graphs, attention cost that grows with context, scheduler jitter |
| n-gram acceptance is a fixed coin flip | Acceptance depends on the text: high on repetitive or copy-heavy output (summaries of the prompt, code edits), low on free-form generation |
| Tensor parallel is one efficiency number | All-reduce over NVLink or PCIe, which dominates on PCIe cards |
| KV cache is always 16-bit | `--kv-cache-dtype` is a separate knob |
| Quality is untouched | FP8 and 4-bit change outputs. Run a quality eval before shipping either |

The qualitative results should reproduce: quantisation helps small batch by shrinking the weight
read and helps large batch by freeing KV room; speculation helps when memory-bound and hurts when
compute-bound; the knee exists. The absolute tokens per second will not match. If a qualitative
result does not reproduce, you have found something worth writing up.

## Files

| File | Purpose |
|---|---|
| `install.sh` | pinned install |
| `serve_bf16.sh`, `serve_fp8.sh`, `serve_awq.sh`, `serve_ngram.sh` | the four configurations |
| `serve_tp2.sh` | optional: the bf16 baseline across 2 GPUs (`--tensor-parallel-size 2`), not run when written |
| `bench_crosscheck.sh` | `vllm bench serve` at one concurrency, to compare with your harness |
| `colab_t4.md` | the free-GPU route |

# Optional extension: a quality-equivalence gate

**Not part of `make check`.** Throughput and cost say a config is faster. They do not say the
answers are still the same. This folder is a gate for that: run the same fixed prompts through a
baseline config and a candidate config, then measure how often the outputs agree.

**Honest scope.** `engine_sim` has no text, so it cannot show a quality change and nothing here
runs against it. The comparator (`check_equivalence.py`) is tested on synthetic outputs only
(`tests/test_given_extras.py`). The recording step (`run_prompts.py`) needs a real `vllm serve`, so
it belongs to the real-GPU session in `gpu/README.md`. It was **not run** while this folder was
written (no GPU on the machine): read the output of your first run before you trust the thresholds.

## The loop (in the GPU session, after the bf16 sweep)

```bash
bash gpu/serve_bf16.sh                                  # terminal 1
python quality/run_prompts.py --name bf16               # terminal 2, writes quality/outputs/bf16.jsonl
# stop the server; start the candidate
bash gpu/serve_ngram.sh                                 # or serve_fp8.sh, serve_awq.sh
python quality/run_prompts.py --name ngram
python quality/check_equivalence.py --baseline quality/outputs/bf16.jsonl \
    --candidate quality/outputs/ngram.jsonl --min-exact 0.9 --min-prefix 0.95
```

`prompts.jsonl` holds 32 fixed prompts in five kinds (fact, math, code, copy, reason). The `copy`
kind matters for speculation: n-gram drafting proposes text it can copy from the prompt, so these
prompts show the case where speculation accepts the most tokens.

## What to write down

| Candidate | What to expect | What a pass does and does not mean |
|---|---|---|
| n-gram speculation | The vLLM docs say speculative decoding is lossless up to hardware numerics, and that greedy sampling with it matches greedy without it | A pass means the engine kept that promise on 32 prompts. A fail is a bug report, not a trade-off |
| fp8 or AWQ weights | Outputs change. How much depends on the model and the method | A pass means "similar on 32 prompts". It is a smoke test, not accuracy: run a real eval on your task before you ship |

You choose `--min-exact` and `--min-prefix`; there is no default pass mark, on purpose. Say in the
report why you chose them. If the check fails, find the first differing word
(`diverge`, and `least similar` in the output) and read both outputs side by side before you
decide whether the change is a drift or a bug.

Source for the speculation row: https://docs.vllm.ai/en/latest/features/speculative_decoding/
(read 2026-11-16).

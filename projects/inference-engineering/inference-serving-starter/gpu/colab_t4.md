# Free GPU route: Colab T4

A T4 (16 GB, compute capability 7.5) is enough to run the full loop with a small model. It is
the cheapest way to see a real knee. It is not representative of an L4 or an H100.

Constraints you must design around:

* No bf16 on Turing: use `DTYPE=half`.
* No FP8 tensor cores: `serve_fp8.sh` runs weight-only FP8 and shows a smaller gain. Say so in your report.
* No FlashAttention on Turing: vLLM picks another attention backend. If startup fails, read the log
  rather than guessing flags.
* 16 GB total: use a 1 to 3B model and `MAX_MODEL_LEN=2048`.
* Colab sessions end. Save `results/*.json` to Drive before the runtime resets.

```python
# Cell 1: install (the pin matches gpu/install.sh)
!pip install -q "vllm==0.31.0"

# Cell 2: start the server in the background
import subprocess, time, httpx
server = subprocess.Popen(
    ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct", "--dtype", "half",
     "--max-model-len", "2048", "--max-num-seqs", "64",
     "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--port", "8000"])
for _ in range(180):
    try:
        if httpx.get("http://127.0.0.1:8000/health", timeout=1).status_code == 200:
            break
    except Exception:
        time.sleep(2)

# Cell 3: clone YOUR fork (it holds your bench/ and analysis/ code), then run the harness
!git clone --depth 1 https://github.com/<your-user>/aiml-companion.git
%cd aiml-companion/projects/inference-engineering/inference-serving-starter
!pip install -q httpx numpy pandas matplotlib
!python -m bench.cli sweep --base-url http://127.0.0.1:8000 --name t4_bf16 --levels 1,2,4,8,16,32 --out results/t4_bf16.json
```

For speculation, restart the server with
`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}'`.

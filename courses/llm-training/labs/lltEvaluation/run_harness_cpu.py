"""Step 1 on a CPU: lm-evaluation-harness 0.4.13 on SmolLM2-360M base and Instruct.

Runs arc_easy, hellaswag and piqa with --limit 500 (the FIRST 500 documents of each split),
log-likelihood scoring, zero-shot, per-item samples logged for the paired analysis.
Usage: python run_harness_cpu.py <python with lm_eval> <output dir>
"""
import os, subprocess, sys, time, json

py, out = sys.argv[1], sys.argv[2]
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
timing = {}
for name in ["HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct"]:
    t = time.time()
    cmd = [py, "-m", "lm_eval", "--model", "hf",
           "--model_args", f"pretrained={name},dtype=float32",
           "--tasks", "arc_easy,hellaswag,piqa", "--limit", "500",
           "--batch_size", "16", "--log_samples", "--seed", "0",
           "--output_path", out]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    timing[name] = round(time.time() - t, 1)
    print(name, timing[name], "s", flush=True)
with open(os.path.join(out, "timing.json"), "w") as f:
    json.dump(timing, f)

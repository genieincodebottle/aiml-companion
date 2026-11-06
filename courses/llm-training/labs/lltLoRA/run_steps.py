"""Run the CPU page cells of lltLoRA in one namespace, in order, and record their printed output.

Usage (from this folder): ..\\..\\.venv-latest\\Scripts\\python.exe run_steps.py
Writes ../../runs/lltLoRA/lltLoRA-t1-cells-v2026-11.json. Exact values only (no timings: see measure_timing.py).
"""
import contextlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = HERE / "steps"
sys.path.insert(0, str(STEPS))
os.chdir(STEPS)
CELLS = ["s01_lora_linear.py", "s02_merge.py", "s03_inject_gpt.py", "s04_peft_check.py"]
if len(sys.argv) > 1:
    CELLS = sys.argv[1:]


def peak_rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 1e9, 2)
    except Exception:
        return None


def main():
    import numpy, torch, transformers, tokenizers, huggingface_hub, peft
    ns = {"__name__": "__main__"}
    samples = []
    for cell in CELLS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile((STEPS / cell).read_text(encoding="utf-8"), cell, "exec"), ns)
        print("=====", cell)
        print(buf.getvalue())
        samples.append({"cell": cell, "stdout": buf.getvalue()})
    run = {
        "runId": "lltLoRA-t1-cells-v2026-11", "module": "lltLoRA", "tier": "T1", "label": "baseline",
        "hardware": {"gpu": None, "count": 0, "vramGB": None, "cpu": "AMD Ryzen 9 5900HS (8 cores, 16 threads), 32 GB RAM"},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__,
                "peft": peft.__version__, "numpy": numpy.__version__, "os": platform.platform()},
        "notebookCommit": None, "config": {"cells": CELLS, "threads": 4}, "seed": None,
        "metrics": [{"step": 0, "peakRssGB": peak_rss_gb()}], "evals": {}, "samples": samples,
        "date": time.strftime("%Y-%m-%d"),
    }
    out = HERE.parent.parent / "runs" / "lltLoRA"
    out.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 1:
        (out / "lltLoRA-t1-cells-v2026-11.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
        print("wrote", out / "lltLoRA-t1-cells-v2026-11.json")


if __name__ == "__main__":
    main()

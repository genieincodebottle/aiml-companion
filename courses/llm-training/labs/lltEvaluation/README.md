# lltEvaluation: Evaluating Trained Models Like a Researcher

LLM Training Lab, module 21. Regime: INFERENCE & EVAL (two short fine-tunes are used as test subjects).

Open `lltEvaluation.ipynb` in Colab or Kaggle with a T4. Steps 1 to 5 also run on a CPU, where the notebook switches to
SmolLM2-360M and SmolLM2-360M-Instruct with 200 items per task.

## Files

| File | What it is |
|---|---|
| `lltEvaluation.ipynb` | The lab notebook, built from the page by `build_notebook.mjs` (runtime check, pinned install, then cells c01 to c11) |
| `notebook_head.py` | The runtime check and the pinned install cells |
| `cells/c01..c11` | The page's code, one file per notebook cell. The Kaggle kernels run these files |
| `browser/challenge_*.py` | The in-page challenge (Pyodide, NumPy only): a paired bootstrap |
| `worker_eval.py` | The experiment queue of kernel llt-eval-main v1 (extra models, seed 1, the safety-mix arm, the GSM8K diagnosis) |
| `run_cells.py` | Runs the cells outside a notebook (CPU smoke and the T1 run) |
| `record.py`, `record_main.py`, `record_extra.py`, `record_handcheck.py`, `record_audit_extras.py`, `record_peak.py` | Write the Run Library files in `../../runs/lltEvaluation/` |
| `measure_timing.py` | T1 timings, run on a quiet machine |
| `make_module_code.py`, `check_cells.py` | Keep the page, the notebook and the cells identical |
| `poll_kernel.py` | Waits for a Kaggle kernel and downloads its output |
| `../../kaggle/cells_eval_peak.py` | Audit-pass kernel llt-eval-peak: per-cell peak GPU memory and the float32 softmax check |
| `scorer.py`, `stats.py`, `refusal.py`, `calibrate_refusal.py`, `measure_scoring.py`, `run_harness_cpu.py` | Earlier helpers (2026-11-05); the T0 refusal calibration run came from `calibrate_refusal.py` |

## Data and safety scope

- XSTest (Röttger et al. 2024, CC-BY-4.0) is read from the authors' GitHub release at commit `d7bb5bd`. The Hub copy
  `walledai/XSTest` is gated (Hub login and one click). HarmBench and StrongREJECT are gated the same way on the Hub; this lab does not use them.
- Measurement only, with established benchmark prompts. No harmful data is generated or trained on. The kernels keep no
  reply to an unsafe prompt beyond its first six words, used to check refusal labels by hand. The run files hold refusal flags, ids, counts of opener types and the first 200 characters of each reply to a SAFE prompt; no reply to an unsafe prompt.
- The fine-tunes are benign: Dolly 15k (CC-BY-SA-3.0). The safety-mix arm adds refusals that Qwen3-1.7B wrote to
  first human turns of hh-rlhf harmless-base (MIT; its card warns of offensive content); their texts were not stored.

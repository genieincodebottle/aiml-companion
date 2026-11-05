# lltMemoryThroughput - Make Training Fit: Memory, Speed and Two GPUs

LLM Training Lab, module 8. Regime: FROM SCRATCH (systems). Hardware: Kaggle, Settings > Accelerator > GPU T4 x2.
Parts 1 and 3 also run on one Colab T4. Part 2 needs two GPUs.

## Files

| File | What it is |
|---|---|
| `lltMemoryThroughput.ipynb` | The learner notebook, built by `build_notebook.py` from `steps/` |
| `steps/` | Every code block on the lab page, verbatim. The page and the notebook are built from these files |
| `challenges/` | The two in-page Prove it challenges (NumPy only), starter and solution |
| `kaggle_cells_p1.py`, `_p2.py`, `_p3.py` | The measurement kernels. Each runs the step files exactly as shipped and records their stdout |
| `make_runs.py` | Turns the kernel outputs (`../../kaggle/out/llt-memthru-p*/`) into Run Library files in `../../runs/lltMemoryThroughput/` |
| `measure.py` | Rebuilds the three kernels and prints the push commands |
| `module/` | The lab page source, assembled by `module/build_module.py` |

## Reproduce our runs

```
python measure.py                      # builds kaggle/llt-memthru-p1, -p2, -p3
# push each with: kaggle kernels push -p kaggle/<slug> --accelerator NvidiaTeslaT4  (private)
# then: kaggle kernels output genieincodebottle/<slug> -p kaggle/out/<slug>
python make_runs.py
```

Measured on Kaggle, 2x Tesla T4 15.6 GB, Python 3.13.16, torch 2.11.0+cu130, 2026-11-05.

## Notes

- Pick the dtype by `torch.cuda.get_device_capability() >= (8, 0)`. `torch.cuda.is_bf16_supported()` returns True on a T4.
- Restart the kernel between part 1 and part 2, so part 1's tensors do not hold GPU 0's memory.
- Step 10 writes a 16.2 GB sharded checkpoint to `/tmp` and deletes it at the end of the notebook.

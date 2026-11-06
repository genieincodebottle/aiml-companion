# lltTrainingLoop, The Training Loop, Line by Line

LLM Training Lab, module 5. Regime FROM SCRATCH. Char-level Tiny Shakespeare.

## Run it on a laptop

Use Python 3.11 or later (3.12 recommended). In a fresh virtual environment:

```
pip install torch numpy jupyter
jupyter notebook lltTrainingLoop.ipynb
```

Run the cells in order. The notebook needs no other package. Step 1 downloads Tiny Shakespeare (1.1 MB) into `data/`. Steps 1 to 8 and 10 run on the CPU; step 9 needs a GPU, so read our runs for it or run it on a free T4.

## Files

| File | What it is |
|---|---|
| `lltTrainingLoop.ipynb` | The lab notebook. Its cells are the page's code blocks, in order. The first cell checks the runtime. |
| `steps/*.py` | The page's code blocks as files. `sN_*.py` define, `sN_run.py` run. `ch1_*`, `ch2_*` are the in-page challenges (NumPy only). |
| `run_steps.py` | Runs code blocks in order in one namespace, the way the notebook does. |
| `measure.py` | CPU (T1) Run Library measurements, deterministic mode, 4 threads. Groups `cosine`, `wsd`, `lrhigh`, `accum`, `resume`, `adamw`. |
| `measure_timing.py` | Every timing figure the page quotes. Run it alone on an idle machine. It prints one JSON line. |
| `runlog.py` | Writes one Run Library JSON (plan section 7.6) to `../../runs/lltTrainingLoop/`. |
| `../../kaggle/cells_trainingloop.py` | The T2 sweep on Kaggle's 2x T4, one worker process per GPU, built with `../../kaggle/build_nb.py`. |

## Reproduce

```
python run_steps.py work steps/s1_data.py steps/s1_run.py steps/s2_model.py steps/s2_run.py steps/s3_sched.py steps/s3_run.py steps/s4_run.py steps/s5_ckpt.py steps/s5_run.py steps/s6_train.py steps/s6_run.py steps/s6_sample.py steps/s7_run.py
python measure.py adamw resume cosine wsd accum lrhigh
python measure_timing.py
```

The T2 sweep runs on Kaggle: `python ../../kaggle/build_nb.py ../../kaggle/cells_trainingloop.py ../../kaggle/llt-trainingloop-sweep llt-trainingloop-sweep "LLT Trainingloop Sweep"`, then push with `--accelerator NvidiaTeslaT4`.

## Notes

- The exact-resume test is a CPU test with `torch.use_deterministic_algorithms(True)`. Some CUDA kernels can change the last bits between runs, so a GPU resume is close but not bit-identical.
- The T4 has compute capability 7.5. `pick_dtype()` chooses fp16 there by capability, because `torch.cuda.is_bf16_supported()` returns True on a T4 too.
- On Kaggle, each worker process sets `CUDA_VISIBLE_DEVICES` before it imports torch, so each run sees one GPU.

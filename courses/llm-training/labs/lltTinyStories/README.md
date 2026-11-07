# lltTinyStories: Your First Real Pretraining Run

LLM Training Lab, module 7. Regime: FROM SCRATCH. Random weights, every weight changes, objective next-token NLL.

Open the notebook in Colab (Runtime > Change runtime type > T4 GPU):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltTinyStories/lltTinyStories.ipynb

With no GPU the same notebook runs the CPU config (T1, a 3.0M-parameter model). Set `FROM_OUR_CHECKPOINT = True` in the config cell to skip the CPU training: it loads our 11.5M step-2,000 model and only evaluates and samples it (val loss 1.7323, 3.4 min on our laptop; run `runs/lltTinyStories/lltTinyStories-T1-from-our-checkpoint.json`).

## Files

| File | What it is |
|---|---|
| `lltTinyStories.ipynb` | The notebook, built from `cells_lab.py` by `build_lab_nb.py`. Its cells are the lab page's code cells. |
| `cells_lab.py` | The notebook cells. The first code cell checks the runtime and picks the tier and dtype. |
| `run_cells.py` | Runs the cells in order on this machine (T1) and writes `runs/lltTinyStories/lltTinyStories-T1-cpu-*.json`. |
| `model_def.py` | The same GPT class, for the CPU checks. |
| `measure.py` | Exactness checks on a CPU: `tokleak`, `bpb`, `prompts`, `resumecheck`. |
| `measure_timing.py` | Every timing figure on the page for T1. Run it on an idle machine. |
| `scaling_fit.py` | The IsoFLOP fit from the page (NumPy only): `synth` recovers a known exponent, `ours` runs on our runs. |
| (Hugging Face Hub) | Our baseline weights at step 2,000 and its tokenizer, for "start from our checkpoint", are `step2000.pt` and `tokenizer.json` in the model repo `aimlcompanion/llm-training-lab`, folder `lltTinyStories/`: https://huggingface.co/aimlcompanion/llm-training-lab/tree/main/lltTinyStories. They are not in this git repo; `../../hf_upload/upload_lltTinyStories.py` uploads them. |

The GPU runs come from two Kaggle kernels in `../../kaggle/`: `llt-tinystories-pretrain-test` (`cells_tinystories.py`, the seed-0 baseline) and `llt-tinystories-sweep` (`cells_tinystories_sweep.py` with `prep_tinystories.py` and `worker_tinystories.py`: sizes, the second seed, depth vs width, learning rate and the resume test). The sweep encodes the corpus once, caches the ids as uint16 `.npy` files, and trains two queues at once, one per T4.

## Data

`roneneldan/TinyStories` (Eldan and Li 2023), default config: 2,119,719 training and 21,990 validation stories. CDLA-Sharing-1.0. The tokenizer trains on the first 300,000 training stories only.

## Reproduce

```
pip install torch "tokenizers==0.23.3" "datasets==5.1.0" numpy
python run_cells.py            # T1 CPU run
python measure.py bpb          # needs the step-3,000 checkpoint; set LLT_CKPT_DIR to the folder that holds it
python scaling_fit.py synth
```

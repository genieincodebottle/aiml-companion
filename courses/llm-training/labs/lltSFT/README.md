# lltSFT: Supervised Fine-Tuning From Scratch, Then With TRL

LLM Training Lab, module 11. Regime: SFT. Every weight of a base model changes; the objective is the negative
log-likelihood of the assistant tokens only. Every figure on the lab page comes from a file in
`courses/llm-training/runs/lltSFT/`.

Open the notebook in Colab (steps 1 to 7 run on the CPU runtime; step 8 needs a T4 GPU runtime):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltSFT/lltSFT.ipynb

## Files

| File | What it is | Where it runs |
|---|---|---|
| `lltSFT.ipynb` | The notebook: a runtime check, the pinned install, a cell that writes `sft_common.py`, then the lab page's code cells in order | Colab or Kaggle T4 (steps 1 to 7 also on a laptop CPU) |
| `sft_common.py` | The data selection, the training chat template, the label mask and the two evaluation helpers (held-out loss, greedy stop rate) | everywhere |
| `worker_sft.py` | The Run Library runs: full SFT of SmolLM2-360M with TRL, one config after another on one GPU | Kaggle kernel `llt-sft-runs` (2x T4, one queue per GPU; queues in `kaggle/cells_sft_runs.py`) |
| `measure.py` | Exact CPU checks: `data`, `template`, `masktrl`, `trlpack`, `packing`, `truncation` | laptop CPU |
| `run_steps.py` | Runs the CPU page cells (steps 1 to 7) in one process and records run `lltSFT-t1-cells` | laptop CPU |
| `measure_timing.py` | The page's CPU timing figures, run on an idle machine. Prints one JSON line | laptop CPU |
| `record_gpu.py` | Copies the Kaggle outputs into the Run Library | any |
| `tryit_data.py` | Writes the Try it widget data from run `lltSFT-t1-packing` (no new measurement) | any |
| `build_notebook.mjs` | Builds the notebook and `steps/` from the module draft, so the page and the notebook cannot disagree | node |
| `runlog.py` | Writes one Run Library JSON file (plan section 7.6 schema) | any |

## Models, data and licences

| Item | Use | Licence |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` and `-Instruct` | Steps 1 to 7 on a CPU; the instruct tokenizer lends its chat template | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-360M` and `-Instruct` | Step 8 and the Run Library; the instruct model is the reference row | Apache-2.0 |
| `HuggingFaceTB/smoltalk2`, config `SFT`, splits `smoltalk_smollm3_smol_magpie_ultra_no_think`, `smoltalk_smollm3_smol_rewrite_no_think`, `smoltalk_smollm3_smol_summarize_no_think` | 3,600 training and 300 held-out conversations of at most 1,024 tokens | Apache-2.0 per the SmolTalk card (Smol-Magpie-Ultra, Smol-rewrite, Smol-summarize). Smol-Magpie-Ultra was generated with Llama-3.1-405B-Instruct; read the Llama 3.1 licence terms before you publish a model trained on it |

SmolTalk2 (created 2025-07-10, last modified 2025-10-31, checked 2026-11-06) is the post-training mix of SmolLM3. It
stores each system prompt in `chat_template_kwargs.custom_instructions`, not in `messages`; `to_messages` moves it back.
The selection streams the splits in file order and keeps the first 1,300 conversations of each split that render to at
most 1,024 tokens: the first 100 are held out, the next 1,200 train. Hashes of both sets are in `lltSFT-t1-data`.

## Reproduce

```
pip install torch "transformers==5.22.1" "trl==1.17.1" "accelerate==1.15.0" "datasets==5.1.0" "tokenizers==0.23.3" "huggingface_hub==2.4.0" numpy psutil
python measure.py data            # writes data_cache.json, the other commands reuse it
python measure.py template
python measure.py masktrl
python measure.py trlpack
python measure.py packing
python measure.py truncation
node build_notebook.mjs <path to lltSFT.mjs>
python run_steps.py
```

The GPU runs: build the kernel with `python build_nb.py cells_sft_runs.py llt-sft-runs llt-sft-runs "LLT SFT Runs"` in
`courses/llm-training/kaggle/`, push it with `--accelerator NvidiaTeslaT4`, download its output, then
`python record_gpu.py <output dir>`.

T1 figures: Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, trl 1.17.1, accelerate 1.15.0, datasets 5.1.0,
tokenizers 0.23.3, huggingface_hub 2.4.0, numpy 2.5.4, on an AMD Ryzen 9 5900HS laptop CPU, 4 threads.
T2 figures: the Kaggle image of the run (Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.1, trl 1.17.1,
accelerate 1.15.0, datasets 5.1.0; versions in each run file), one Tesla T4, fp16 autocast with fp32 master weights.

## Kaggle validation

- `llt-sft-runs` (2026-11-06, one 2x T4 session of 6,431 s): the 11 `lltSFT-T2-*` runs, two queues, one per GPU.
- `llt-sft-notebook`: this notebook end to end on one T4, built by `kaggle/make_sft_notebook_kernel.py`, which swaps the
  Prove it starter for its solution and appends two recording cells (the step 8 figures and an SDPA kernel probe).

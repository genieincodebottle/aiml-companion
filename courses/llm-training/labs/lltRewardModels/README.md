# lltRewardModels: Preference Data and Reward Models

LLM Training Lab, module 17. Regime: PREFERENCE. SmolLM2-360M gets a scalar head and every weight trains with the
Bradley-Terry loss on UltraFeedback preference pairs. Every figure on the lab page comes from a file in
`courses/llm-training/runs/lltRewardModels/`.

Open the notebook in Colab (steps 1 to 4 run on a CPU runtime; step 5 on needs a T4 GPU runtime):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltRewardModels/lltRewardModels.ipynb

## Files

| File | What it is | Where it runs |
|---|---|---|
| `lltRewardModels.ipynb` | The notebook: a runtime check, the pinned install, then the lab page's code cells in order | Colab or Kaggle T4 (steps 1 to 4 also on a laptop CPU) |
| `steps/cellNN_stepK.py` | The page's code cells, one file each. The notebook and the page are built from them | everywhere |
| `build_notebook.py` | Builds the notebook and the Kaggle kernel `llt-rm-notebook` from `steps/` | any |
| `export_page_code.py` | Puts the same cells and the challenge into the lab page draft | any |
| `run_steps.py` | Runs steps 1 to 4 on a CPU in one process and records run `lltRewardModels-t1-cells` | laptop CPU |
| `measure.py data` | Length and tie statistics of the full preference splits, run `lltRewardModels-t1-data` | laptop CPU |
| `measure.py trlfilter` | How many pairs RewardTrainer keeps at `max_length` 512 and 1,024 on 1,000 pairs, run `lltRewardModels-t1-trlfilter` | laptop CPU |
| `measure_timing.py` | CPU timing of steps 1 to 4 on an idle machine, one JSON line | laptop CPU |
| `record_notebook.py` | Records the notebook kernel run `llt-rm-notebook` as `lltRewardModels-T2-notebook` | any |
| `record_gpu.py` | Turns the Kaggle output of `llt-rm-main` into Run Library files and the best-of-n analysis | any |
| `tryit_data.py` | Writes the Try it widget data from the recorded runs (no new measurement) | any |
| `challenge_solution.py`, `challenge_starter.py` | The in-page NumPy challenge | browser (Pyodide) or CPython |
| `runlog.py` | Writes one Run Library JSON file (plan section 7.6 schema) | any |

The Run Library runs come from the Kaggle kernel `kaggle/llt-rm-main` (cells in `kaggle/cells_rm_main.py`, worker
`kaggle/worker_rm.py`): the best-of-n sample pool, the gold scores, the TRL check and four reward-model runs in one T4
session.

## Models, data and licences

| Item | Use | Licence |
|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` | The reward model's backbone | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | Lends its chat template; the policy sampled for best-of-n (fp32: fp16 sampling crashes on a T4) | Apache-2.0 |
| `Skywork/Skywork-Reward-V2-Qwen3-1.7B` | The gold reward model, scored with its own chat format as on its model card | Apache-2.0 |
| `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs`, `test_prefs`) | 3,000 training pairs (shuffle seed 0) and 1,000 held-out pairs; 64 held-out prompts for best-of-n | MIT. The ratings come from GPT-4, so OpenAI's terms on model outputs may apply to what you build with them |
| `allenai/reward-bench-2` | A 300-item subset (60 per best-of-4 domain, seed 0) | ODC-BY |

`nvidia/HelpSteer3` (CC-BY-4.0) is the alternative preference set named in the plan; this lab does not use it.

## Hub files (model repo `aimlcompanion/llm-training-lab`, folder `lltRewardModels/`)

- `bon_pool.json`: the 64 prompts, 4,096 samples and the gold reward of each sample (step 8 downloads it).
- `rm-s0/`: our seed-0 reward model, fp16 safetensors (step 5 loads it when `RUN_TRAINING = False`). It leans heavily
  on reply length (held-out 0.723 when the chosen reply is longer, 0.496 when shorter).

## Reproduce

```
python run_steps.py                 # steps 1 to 4 on a CPU
python measure.py data              # data statistics
python build_notebook.py            # notebook and Kaggle kernel
python record_gpu.py                # after downloading kaggle/out/llt-rm-main
```

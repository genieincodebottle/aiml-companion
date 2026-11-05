# lltIntro, What It Takes to Train an LLM

LLM Training Lab, module 1 (free). Regime: INFERENCE & EVAL. No weights change in the lab itself.
Every number on the lab page comes from a run in `../../runs/lltIntro/` or from a derivation.

## Files

| File | What it does |
|---|---|
| `lltIntro.ipynb` | The lab notebook. Runs every step of the page on a CPU. Open in Colab or run locally. |
| `story-model.json` | The 209,408-parameter character-level story model that step 1 runs in the browser. float16, base64. |
| `train_story_model.py` | Pretrains that model FROM SCRATCH on TinyStoriesV2-GPT4-valid.txt and exports the JSON. Also produces the broken and control runs. |
| `measure.py` | Step 3 and 4 measurements on SmolLM2-135M, and the initializer_range = 1.0 broken run. |
| `texts.py` | The English, Python and Hindi texts measured in step 4. |
| `runlog.py` | Writes one Run Library JSON per run. |

## Reproduce the Run Library

```
python train_story_model.py --seed 0 --steps 5000 --threads 4 --run-id lltIntro-t1-story-s0 --export story-model.json
python train_story_model.py --seed 1 --steps 5000 --threads 4 --run-id lltIntro-t1-story-s1
python train_story_model.py --seed 2 --steps 5000 --threads 4 --run-id lltIntro-t1-story-s2
python train_story_model.py --seed 0 --steps 300 --eval-every 50 --threads 3 --run-id lltIntro-t1-story-short-s0 --label control:300-steps
python train_story_model.py --seed 0 --steps 300 --eval-every 50 --threads 3 --init-std 1.0 --run-id lltIntro-t1-broken-initstd1 --label broken:init-std=1.0
python measure.py
```

Data: `roneneldan/TinyStories`, file `TinyStoriesV2-GPT4-valid.txt` (licence CDLA-Sharing-1.0, Eldan and Li 2023).
Stories with characters outside letters, space, newline and `. , ! ? ' " - : ;` are dropped after curly quotes and dashes are normalised (27,381 of 27,631 kept). The split is 95/5 by story with shuffle seed 0.

Model: SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`, Apache 2.0, Allal et al. 2025).

Environment of the recorded runs: Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, tokenizers 0.23.3, numpy 2.5.4, AMD Ryzen 9 5900HS, no GPU. The SmolLM2-135M losses, tokens per word and greedy samples were also measured on the earlier pins (torch 2.10.0, transformers 5.2.0, tokenizers 0.22.1, numpy 2.4.2) and are identical to the digit; the `-v2026-11` run files hold the current-pin re-runs. The browser step was checked on Pyodide 0.27.0 and 0.27.8 (numpy 2.0.2 in both, same output).

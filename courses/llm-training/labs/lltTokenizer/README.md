# lltTokenizer: Build a Byte-Level BPE Tokenizer

LLM Training Lab, module 2. Regime: FROM SCRATCH (tokenizer). No model weights change; you build the merge list.

Open the notebook in Colab (CPU runtime is enough):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltTokenizer/lltTokenizer.ipynb

Run the cells in order from the top. The setup cells install the pinned libraries, write `bpe.py` and `data.py`, and download about 60 MB of text. Steps 6 to 10 reuse names from earlier cells (`train_bpe`, `train`, `held`, `ours`), so run them in order.

On a laptop (Python 3.11 or later, 3.12 recommended):

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS and Linux
pip install psutil notebook
jupyter notebook lltTokenizer.ipynb
```

The install cell uses `%pip`, which installs into the running kernel.

## Files

| File | What it is |
|---|---|
| `lltTokenizer.ipynb` | The notebook. Its cells are the lab page's code cells, in order. |
| `bpe.py` | Byte-level BPE in pure Python: GPT-2 pre-tokenization without the `regex` module, training, encode by merge rank, decode, loading GPT-2's merges. Standard library only, so it also runs in Pyodide. |
| `data.py` | Downloads and caches the text (about 60 MB in `~/.cache/lltlab/tokenizer`). |
| `browser/step1.py` to `step4.py` | The cells that run in the lab page (Pyodide 0.27, standard library only, 50 lines or fewer each). |
| `steps/step5.py` to `step10.py` | The cells that need `tokenizers` and the data. They share one namespace, like notebook cells. |
| `measure.py` | Regenerates every Run Library file in `../../runs/lltTokenizer/`. `python measure.py all`. |
| `run_steps.py` | Runs every page cell in order and records the output (`lltTokenizer-t1-cells`). |
| `pyodide_time.mjs` | Times the browser cells in real Pyodide 0.27.0 under Node (`lltTokenizer-t0-pyodide`). |

## Data and licences

| Source | Use | Licence |
|---|---|---|
| FLORES-200 devtest, English, Hindi, Spanish (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`, sha256 checked) | Held-out text; the same 1,012 sentences in each language | CC BY-SA 4.0 |
| Wikipedia 20231101 English, Hindi, Spanish (`wikimedia/wikipedia`), first articles up to 8 MB each | Training text | CC BY-SA 3.0 and GFDL |
| CPython 3.12 standard library | Training code (4 MB) and held-out code (12 files) | PSF License |
| GPT-2, SmolLM2-135M and Qwen3-0.6B `tokenizer.json` from the Hugging Face Hub | Comparison tokenizers | MIT, Apache-2.0, Apache-2.0 |

## Reproduce

```
pip install tokenizers==0.23.3 huggingface_hub==2.4.0 datasets==5.1.0 transformers==5.22.1 torch==2.15.0 psutil
python data.py
python measure.py all
python run_steps.py
```

Recorded on a laptop CPU (AMD Ryzen 9 5900HS, 16 threads, 33.7 GB RAM), Python 3.12.3, tokenizers 0.23.3, huggingface_hub 2.4.0, datasets 5.1.0, transformers 5.22.1, torch 2.15.0+cpu, numpy 2.5.4, no GPU. The run files in `../../runs/lltTokenizer/` were measured on tokenizers 0.22.1, huggingface_hub 1.5.0 and transformers 5.2.0 (each file's `env`). `lltTokenizer-t1-notebook-v2026-11b` is the whole notebook run in a fresh environment on the pins above, and every token count and ratio it prints is identical to those files. The code column depends on the Python version, because the code corpus is the interpreter's own standard library (Python 3.13 on Kaggle gives different code figures). Times depend on the machine and on other load; every run file records the CPU and how busy it was.

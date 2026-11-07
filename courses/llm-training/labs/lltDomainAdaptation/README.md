# lltDomainAdaptation: adapt a model to Hindi with continued pretraining, fine-tuning or RAG

LLM Training Lab, module 15. Regime: continued pretraining, plus a QLoRA (PEFT) step.

- **Model:** Qwen3-0.6B-Base.
- **Domain:** Hindi, from FineWeb2 `hin_Deva` (ODC-By).
- **Task:** XQuAD Hindi (CC-BY-SA-4.0).
- **Panel:** XStoryCloze hi/en (CC-BY-SA-4.0), HellaSwag (Rowan/hellaswag; its Hub card states no licence), and FineWeb-Edu replay text (ODC-By).

## Files

| File | What it is |
|---|---|
| `lltDomainAdaptation.ipynb` | Learner notebook, one T4 (Colab or Kaggle). Built by `make_notebook.py`; its code cells are `page_blocks.py` |
| `dalab.py` | Lab library: data, the evaluation panel (bits per byte, choice accuracy), RAG prompts and QA scoring, continued pretraining loop with re-warm/re-decay, interpolation, mean-of-subword init, the LoRA QA loop |
| `extend_tokenizer.py` | Continued-BPE extension of Qwen3's tokenizer from Devanagari pieces; `pin_special=False` reproduces the shifted special-token ids |
| `page_blocks.py` | The page's code blocks. `export_blocks.py` writes them into the page draft |
| `qa_split.json` | XQuAD hi train/test question ids, 3 example pairs, and e5-small top-5 retrieval for each test question |
| `fertility.py`, `run_extend.py`, `rag_prep.py`, `run_cpu_blocks.py` | CPU measurements (T1 run files) |
| `measure_timing.py` | CPU timings for the serial quiet-machine pass, printed as one JSON line |
| `to_runlib.py` | Converts the Kaggle sweep's `res_*.json` into Run Library files |
| `measure_init.py`, `test_local.py` | `measure_init.py` is superseded by the GPU `panel0`; `test_local.py` is a CPU smoke test of the eval code |

GPU runs: `../../kaggle/make_da_kernel.py` builds the sweep, and `../../kaggle/worker_da.py` runs one job list per GPU. The sweep was `llt-da-sweep` v1 on 2026-11-06, with output in `../../kaggle/out/llt-da-sweep/`. Run files are in `../../runs/lltDomainAdaptation/`.

The notebook's 150-step cell is the quick path. Our recorded runs use 320 steps (2.6M tokens).

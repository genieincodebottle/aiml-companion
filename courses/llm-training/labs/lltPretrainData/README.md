# lltPretrainData, Turn Raw Web Pages Into Pretraining Data

LLM Training Lab, module 9. Every figure on the lab page comes from a file in
`courses/llm-training/runs/lltPretrainData/`.

| File | What it does | Where it runs |
|---|---|---|
| `lltPretrainData.ipynb` | The lab notebook. Steps 1 to 7 on the first 60 MB of one Common Crawl WARC file | Colab or a laptop CPU (T1) |
| `ptd.py` | The filters, PII scrub, decontamination and MinHash dedup, plain Python and NumPy | everywhere |
| `prep_ptd.py` | The full data prep: the funnel on one whole WARC file, the quality classifier, and the token files for the training runs | Kaggle CPU kernel `llt-ptd-prep` (4 cores) |
| `worker_ptd.py` | The training runs: filtered vs unfiltered, canaries, epochs, mixtures, the WSD anneal | Kaggle GPU kernel `llt-ptd-train` (2x T4, one queue per GPU) |
| `measure.py` | Exact checks on small inputs: rule fixtures, the LSH S-curve, paraphrase decontamination, mixture accounting | CPU |
| `measure_timing.py` | The timing figures on the page, measured on an idle machine | CPU |
| `run_steps.py` | Runs steps 1 to 7 in one process and records run lltPretrainData-t1-notebook | CPU |
| `leak_check.py` | Measures the train and held-out leak against the pages the unfiltered model trained on | CPU |
| `record_prep.py`, `record_gpu.py` | Copy the Kaggle outputs into the Run Library | any |
| `build_notebook.py` | Builds the notebook from `steps/`, `browser/` and `ptd.py` | any |
| `browser/` | The in-page cells and the Prove it solutions (NumPy only, Pyodide 0.27) | browser |

The notebook ran end to end on a Kaggle CPU session (kernel `llt-ptd-notebook` v3, run lltPretrainData-t1-notebook-kaggle) with every band check passing.

## Data sources and licences

- Common Crawl CC-MAIN-2024-10, WARC file `CC-MAIN-20240220211055-20240221001055-00000.warc.gz` (1,134,803,864 bytes) and its WET file. Common Crawl Terms of Use; each page stays the copyright of its site. The lab prints counts and short excerpts only.
- C4 `en.noclean` shard `c4-train.00000-of-07168.json.gz` (allenai/c4, ODC-By 1.0, plus the Common Crawl Terms of Use). The unfiltered training corpus.
- FineWeb-Edu Llama 3 annotations, shard 0 (HuggingFaceFW/fineweb-edu-llama3-annotations, ODC-By 1.0). Labels for the quality classifier.
- wikitext-103-raw-v1 (Salesforce/wikitext, CC BY-SA 3.0) and TinyStories (roneneldan/TinyStories, CDLA-Sharing-1.0). The other two mixture domains.
- ARC-Easy test (allenai/ai2_arc, CC BY-SA 4.0) and HellaSwag validation (Rowan/hellaswag, no licence stated on its card or repository, used for evaluation only). Decontamination targets and the HellaSwag eval.
- fastText `lid.176.bin` (CC BY-SA 3.0). Language ID.

## Reproduce

```
pip install "trafilatura==2.3.1" "warcio==1.8.2" "datasketch==2.0.0" "fasttext-numpy2==0.10.4" "scikit-learn==1.9.1"   # keep your NumPy; on Windows use fasttext-wheel 0.9.2 (run_steps.py patches its predict())
python prep_ptd.py out            # 35 min in our Kaggle CPU session (4 cores); writes out/ptd_prep_stats.json and the token files
python worker_ptd.py queue.json out runs_out   # one GPU; the two queues (18 runs, 59 min on 2x T4) are in kaggle/cells_ptd_train.py
python measure.py out/raw/arc_easy_test.parquet out/ptd_prep_stats.json
```

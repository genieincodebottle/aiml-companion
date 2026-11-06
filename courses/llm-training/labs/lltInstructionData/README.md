# lltInstructionData, Curate Instruction Data and Test It Against Bulk Data

LLM Training Lab, module 14. Regime: SFT. You build a pool of instruction examples from an open mixture
(Dolci Instruct SFT), raw chat logs (WildChat-1M) and a local teacher (Qwen3-4B-Instruct-2507, Magpie-style),
clean it, score it with a reward model, select 1,000 examples and train SmolLM2-360M on them against bulk samples.
Every figure on the lab page comes from a file in `courses/llm-training/runs/lltInstructionData/`.

Open the notebook in Colab (T4 GPU runtime):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltInstructionData/lltInstructionData.ipynb

## Files

| File | What it is | Where it runs |
|---|---|---|
| `cells/c01_pool.py` ... `cells/c12_recorded.py` | The page's code cells, the single source. The notebook, the Kaggle kernels and the page are built from them. c10 and c11 are the recorded runs' single experiment cell split in two, so no cell runs over about 35 min; c11 stops with a message if c10 was skipped. c12 (notebook cell 14) compares our recorded per-prompt scores with no GPU | T4 (c12 any) |
| `notebook_head.py` | The runtime check and the pinned install, the notebook's first two cells | T4 |
| `lltInstructionData.ipynb`, `steps/` | The notebook and the cells as files, built by `build_notebook.mjs` from the module file | T4 |
| `browser/` | The Prove it starter and solution (NumPy only, Pyodide 0.27) | browser |
| `worker_insdata.py` | One GPU queue of the full experiment: train an arm with the page's `train_arm`, score IFEval, answer 200 held-out prompts, judge them | Kaggle 2x T4 |
| `record_gpu.py`, `record_data.py`, `record_exp.py` | Copy the Kaggle outputs into the Run Library: teacher bake-off; steps 1 to 7 (counts re-derived from data/ and checked against the printed log); step 9 runs, paired bootstrap intervals and the notebook record | any |
| `measure_lengths.py` | Answer lengths, the share of examples past max_length 1,024, and reward vs length (run lltInstructionData-T1-length-check) | CPU |
| `make_tryit_data.py` | The threshold sweep and candidate pairs for the Try it widget | CPU |
| `check_cells.py` | Checks that the built notebook's page cells equal cells/*.py | any |
| `measure_embedders.py` | The broken run: an English-only embedder on a multilingual pool | CPU |
| `make_module_code.py` | Writes the cells into the module draft as code constants | any |
| `run_cells.py` | Runs cells in one namespace and records their output; `LLT_SMOKE=1` runs a tiny CPU smoke with `dev/smoke_subs.json` | CPU |
| `runlog.py` | Writes one Run Library JSON file (plan section 7.6 schema) | any |

Kaggle kernels (private, folders under `courses/llm-training/kaggle/`; `llt-insdata-head` is built but not pushed, it checks the rewritten head cells on the image): `llt-insdata-teacher` v2 (teacher bake-off,
`cells_insdata_teacher.py`), `llt-insdata-notebook` v1 (cells 1 to 7 completed, then out of memory at the first
IFEval pass; fixed in the cells) and `llt-insdata-exp` v3 (`cells_insdata_exp.py`: the notebook end to end on GPU 0,
the second seeds and bulk 4,000 on GPU 1). Steps 1 to 7 wrote byte-identical data in v1 and v3.

The recorded runs used an earlier form of the first two cells (torch imported in the runtime check, `!pip`). The
current `notebook_head.py` reads versions with importlib.metadata and installs with `%pip`, with the same pins.

## Data, models and licences

| Item | Use | Licence or terms |
|---|---|---|
| allenai/Dolci-Instruct-SFT, shard 0, revision bd3c8f3a | 4,000 rows of an open SFT mixture | ODC-BY; research and educational use per Ai2's guidelines; per-source licences on the card |
| allenai/WildChat-1M, shard 0 (and shard 1 for held-out prompts), revision 7d6490e4 | 4,000 raw user conversations | ODC-BY; responses from GPT-3.5 and GPT-4, subject to OpenAI's terms |
| Qwen/Qwen3-4B-Instruct-2507 | Teacher | Apache-2.0 |
| ibm-granite/granite-embedding-311m-multilingual-r2 | Near-dup and paraphrase checks | Apache-2.0 |
| Skywork/Skywork-Reward-V2-Qwen3-0.6B | Quality filter | Apache-2.0 |
| Skywork/Skywork-Reward-V2-Llama-3.2-1B | Judge for the held-out prompts (experiment only) | Llama 3.2 Community License |
| dslim/bert-base-NER | Person names in user prompts | MIT |
| nvidia/HelpSteer2 validation | Human labels to check the reward model | CC-BY-4.0 |
| google/IFEval, openai/gsm8k test, openai/openai_humaneval, cais/mmlu test | Decontamination targets; IFEval is also the eval | Apache-2.0, MIT, MIT, MIT |
| HuggingFaceTB/SmolLM2-360M (+ -Instruct for its chat template) | The model trained | Apache-2.0 |

The data pool (`data/`) is written by the notebook and is not part of this repository.

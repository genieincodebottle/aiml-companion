# lltShip: merge, compress and serve your fine-tunes

LLM Training Lab, module 22. Regime POST-HOC.

You merge two of your earlier fine-tunes of Qwen3-0.6B-Base, the instruction SFT model from lltDPO and the Hindi
continued-pretraining model from lltDomainAdaptation, with task arithmetic, TIES and DARE written in plain PyTorch, and
match mergekit 0.1.5 tensor for tensor. Then you compress the merge with GGUF k-quants (llama.cpp b12182), GPTQ W4A16
(llm-compressor 0.15.0) and quantization-aware training, and serve it with llama.cpp and vLLM 0.33.0.

| File | What it is |
|---|---|
| `lltShip.ipynb` | The learner notebook (built by `build_notebook.py`, then `tools/add_run_help.py`) |
| `steps/cellNN_stepK.py` | The page's code cells; the notebook and the page are built from them |
| `challenge_starter.py`, `challenge_solution.py` | The in-page TIES challenge (NumPy) |
| `run_steps.py` | CPU cells (steps 1 to 3 and 6) on a laptop; writes `lltShip-t1-cells` and `lltShip-t1-mergekit` |
| `check_torchao.py` | The page's fake quantization against torchao's; writes `lltShip-t1-torchao-check` |
| `run_gpu.py` | Every cell in one process on a GPU machine (`--mode smoke`, `full`, `defaults`) |
| `record_gpu.py` | Turns `run_gpu.py` output into Run Library files in `runs/lltShip/` |
| `measure_timing.py` | CPU timing for the page on an idle laptop |
| `export_page_code.py` | Writes the page draft's code constants from `steps/` |
| `recorded.json` | Our A100 results for the long steps (5 KB). The notebook downloads it from the Hub (`lltShip/recorded.json`) when `FROM_OUR_CHECKPOINT = True`; built by `make_recorded.py` from `runs/lltShip/`, pushed by `upload_recorded.py` |
| `record_t4.py`, `run_extras_t4.py` | The 2026-11-11 T4 timing runs: the notebook with the exit on and off, and the two optional switches (`RUN_GRID`, `RUN_VLLM`) |
| `tryit_data.py` | Builds `tryit_data.json` for the optional Try it widget (`TRYIT_SPEC.md`) |

Our recorded GPU runs used a Google Cloud A100 40 GB. The learner path is a free T4 (Colab or Kaggle) in fp16, with
merging and GGUF work on the CPU of the same session.

`FROM_OUR_CHECKPOINT = True` (near the top of code cell 1, and the default) is the checkpoint exit for a free T4. Step 5 scores our
shipped merge and reads the other eight merges from our grid, step 7 loads our Q4_K_M file from the Hub, and steps 9 and 12 print our
QAT and vLLM runs. Every later step still runs. Set it to `False` to run every long step yourself. Timed on a Google Cloud T4 VM set up
like Kaggle's GPU T4 x2 (4 vCPUs, 26 GB RAM): 34.0 min with the exit on, 75.5 min with it off (runs `lltShip-T2-gcpT4-notebook` and
`lltShip-T2-gcpT4-notebook-nocheckpoint`).

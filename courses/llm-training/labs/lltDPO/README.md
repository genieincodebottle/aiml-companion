# lltDPO: DPO, derive it, implement it, stress it

LLM Training Lab, module 18. Regime PREFERENCE: a LoRA adapter on an SFT'd Qwen3-0.6B-Base, trained with the DPO loss on
UltraFeedback preference pairs against a frozen reference.

## Files

| File | What it is |
|---|---|
| `lltDPO.ipynb` | The learner notebook. Built by `build_notebook.py` from `steps/`; every code cell is a code block on the lab page |
| `steps/cellNN_stepK.py` | The page code cells, one file each. Steps 1 to 3 run on a CPU, steps 4 to 11 on one T4 |
| `challenge_starter.py`, `challenge_solution.py` | The in-page NumPy challenge (DPO, IPO and SimPO losses, DPO gradients) |
| `build_notebook.py` | Builds the notebook and the Kaggle kernels `llt-dpo-smoke`, `llt-dpo-core`, `llt-dpo-exp` in `../../kaggle/` |
| `run_steps.py` | Runs steps 1 to 3 on the CPU and writes `runs/lltDPO/lltDPO-t1-cells.json` |
| `measure_timing.py` | CPU timing for steps 1 to 3 (run on a quiet machine) |
| `record_gpu.py` | Turns the kernels' output JSON into Run Library files in `runs/lltDPO/` |
| `poll_kernel.py` | Waits for a Kaggle kernel and downloads its output |

## Recorded runs

Kernels run on one Kaggle T4. `llt-dpo-core` trains the SFT model, the main DPO run (beta 0.1, seed 0) and the beta sweep.
`llt-dpo-exp` runs the learner defaults (our SFT model and adapter from the Hub) plus the loss-variant and on-policy steps.

Checkpoints on the Hugging Face Hub, `aimlcompanion/llm-training-lab`:
- `lltDPO/sft-s0/`: Qwen3-0.6B-Base after LoRA SFT on 3,000 UltraFeedback chosen replies, adapter merged, fp16.
- `lltDPO/dpo-b0.1-s0/`: the LoRA adapter of the main DPO run, on top of `sft-s0`.

## Template

Qwen3-0.6B-Base's chat-control tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`) share one untrained embedding
row, so this lab uses a plain-text template and ends every reply with `<|endoftext|>`, a trained token:

```
### Instruction:
{prompt}

### Response:
{reply}<|endoftext|>
```

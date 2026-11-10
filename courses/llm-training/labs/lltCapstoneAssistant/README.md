# Capstone: Build an Instruction-Following Assistant (lltCapstoneAssistant)

The LLM Training Lab capstone for fine-tuning and alignment. Start from the base model `Qwen/Qwen3-1.7B-Base` and finish with an assistant on the Hugging Face Hub:

data card -> SFT (LoRA) -> on-policy preference pairs labelled by a reward model -> DPO (LoRA) -> IFEval, a calibrated open-model judge against `Qwen/Qwen3-1.7B` (raw and length-controlled), an XSTest behaviour panel -> model card -> release.

| File | What it is |
|---|---|
| `RUNBOOK.md` | How to run it: the free T4 path (T2), one cloud GPU (T3), and full fine-tuning (T4) |
| `RUBRIC.md` | The 100-point rubric and the submission checklist |
| `capstone/` | One module per stage, `python -m capstone.<stage> --tier T2|T3 --work <dir>`; every stage resumes |
| `tests/test_cpu.py` | CPU tests for the parts that fail silently: label mask, end-of-text label, packing, pair rule, verdict parser, statistics |
| `gcp/` | Environment setup and the resumable chain for one cloud GPU |
| `notebooks/` | The T2 notebook and the script that builds it |

The reference run (T3, one A100 40 GB) and its run files are in `../../runs/lltCapstoneAssistant/`. The released model is at `aimlcompanion/llm-training-lab`, folder `lltCapstoneAssistant/`.

Data sources keep their own licences, listed in `RUNBOOK.md` and in the model card.

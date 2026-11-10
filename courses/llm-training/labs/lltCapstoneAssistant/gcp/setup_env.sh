#!/bin/bash
# Build the two venvs of the lltCapstoneAssistant reference run on a cloud GPU VM (no GPU needed for this script).
#   ~/capstoneP2/venv-train : torch 2.11.0+cu130 and the track's HF pins (SFT, DPO, reward-model labelling)
#   ~/capstoneP2/venv-vllm  : vLLM 0.33.0 (it pins torch 2.14.0 and transformers < 5.21) for fast generation and the judge,
#                              plus lm-eval 0.4.13 for the IFEval instruction checkers
# Two venvs because vLLM's pins conflict with the training pins (checked on PyPI 2026-11-10).
set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/capstoneP2
[ -d venv-train ] || uv venv venv-train --python 3.12
VIRTUAL_ENV=$HOME/capstoneP2/venv-train uv pip install --python venv-train/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python venv-train/bin/python -r requirements-train.txt
[ -d venv-vllm ] || uv venv venv-vllm --python 3.12
uv pip install --python venv-vllm/bin/python -r requirements-vllm.txt
venv-vllm/bin/python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]"
venv-train/bin/python -c "import sys, torch, transformers, trl, peft, bitsandbytes; print('train', sys.version.split()[0], torch.__version__, transformers.__version__, trl.__version__, peft.__version__, bitsandbytes.__version__)"
venv-vllm/bin/python -c "import sys, torch, transformers, vllm, lm_eval; print('vllm', sys.version.split()[0], torch.__version__, transformers.__version__, vllm.__version__, lm_eval.__version__)"
echo SETUP DONE

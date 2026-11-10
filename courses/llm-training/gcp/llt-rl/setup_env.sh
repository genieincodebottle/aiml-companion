#!/bin/bash
# Build ~/llt-venv (Python 3.12, torch 2.11.0+cu130 as on Kaggle, notebook HF pins). Logs to ~/setup_env_rl.log.
# Copy of tmp/llt-handoff/gcp/setup_env.sh with the requirements path of the lltRL job.
set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/llt-venv ] || uv venv ~/llt-venv --python 3.12
source ~/llt-venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r ~/llt-rl/requirements.txt numpy pandas
python -c "import sys, torch, transformers, trl, peft, bitsandbytes, numpy, math_verify; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, trl.__version__, peft.__version__, bitsandbytes.__version__, numpy.__version__)"
echo SETUP DONE

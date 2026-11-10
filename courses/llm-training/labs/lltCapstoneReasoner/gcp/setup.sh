#!/bin/bash
# One-time setup on the VM. Reuses ~/llt-venv (Python 3.12, torch 2.11.0+cu130, transformers 5.22.1, trl 1.17.1,
# peft 0.21.2, math-verify 0.9.0), the same pins as the lltRL lab, because the capstone runs lltRL's GRPO code.
set -e
mkdir -p ~/capstoneP3/logs ~/capstoneP3/lltRL_steps ~/capstoneP3/data ~/capstoneP3/runs ~/capstoneP3/evals
cd ~/capstoneP3
tar xzf p3_code.tgz
source ~/llt-venv/bin/activate
python -c "import torch, transformers, trl, peft; print(torch.__version__, transformers.__version__, trl.__version__, peft.__version__, torch.cuda.get_device_name(0))"
LLTRL_STEPS=~/capstoneP3/lltRL_steps python test_capstone.py

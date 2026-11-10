#!/bin/bash
# One-time environment for the reference run on a fresh GPU VM (Deep Learning VM image, CUDA 13 driver).
#  ~/llt-venv : Python 3.12, torch 2.11.0+cu130 (the Kaggle image's torch), transformers 5.22.1, trl 1.17.1, peft 0.21.2
#  ~/vllm-venv: vllm 0.33.0 (pins torch 2.14.0), for evaluate_vllm.py only
set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/llt-venv ] || uv venv ~/llt-venv --python 3.12
[ -d ~/vllm-venv ] || uv venv ~/vllm-venv --python 3.12
( source ~/vllm-venv/bin/activate && uv pip install vllm==0.33.0 > ~/vllm_install.log 2>&1 ) &
source ~/llt-venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0 math-verify==0.9.0 safetensors psutil numpy
wait
python -c "import sys, torch, transformers, trl, peft; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, trl.__version__, peft.__version__)"
~/vllm-venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)"
echo SETUP DONE

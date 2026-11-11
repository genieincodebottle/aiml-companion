set -x
date +%s > /tmp/setup_start
sudo mkdir -p /kaggle/working && sudo chown -R $USER /kaggle/working
sudo apt-get install -y -q python3-pip >/dev/null 2>&1
pip install --user -q uv 2>&1 | tail -1
export PATH=$HOME/.local/bin:$PATH
T0=$(date +%s)
uv venv --seed --python 3.12 $HOME/venv 2>&1 | tail -2
. $HOME/venv/bin/activate
python --version
T1=$(date +%s); echo "venv_sec $((T1-T0))"
uv pip install -q torch==2.11.0 numpy jupyter nbconvert ipykernel 2>&1 | tail -3
T2=$(date +%s); echo "torch_jupyter_sec $((T2-T1))"
python -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count())"

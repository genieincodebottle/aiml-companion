"""The two cells every lltEvaluation notebook starts with: the runtime check and the pinned install.
build_notebook.mjs reads them from here as text, so the notebook and this file cannot disagree.
The Kaggle kernels (kaggle/cells_eval.py) run these same two cells first.
"""

RUNTIME_CHECK = '''# Runtime check. Run this cell first. It imports nothing heavy, so the install cell can still change versions.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")   # one GPU per process
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less fragmentation when models come and go
import platform, shutil, subprocess, sys
from importlib.metadata import version, PackageNotFoundError
ok = True
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    ok = False
    print("Python 3.11 or later is needed, and 3.12 is recommended.")
for pkg in ["torch", "numpy", "transformers", "lm_eval"]:
    try:
        print(pkg, version(pkg))
    except PackageNotFoundError:
        print(pkg, "not installed yet")
free = shutil.disk_usage(".").free / 1e9
print("Free disk", round(free, 1), "GB")
if free < 15:
    ok = False
    print("The models in this lab need about 12 GB of disk. Free some space.")
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
except FileNotFoundError:                  # a CPU runtime has no nvidia-smi at all
    gpu = subprocess.CompletedProcess([], 1, "", "")
if gpu.returncode != 0 or not gpu.stdout.strip():
    print("No GPU. Steps 1 to 5 run on a CPU with SmolLM2-360M. Steps 6 to 10 need a T4: in Colab use Runtime > Change runtime type > T4 GPU.")
else:
    print("GPU", gpu.stdout.strip().splitlines()[0])
    if "P100" in gpu.stdout:
        ok = False
        print("Current PyTorch has no kernels for the P100. Switch the accelerator to T4.")
print("Ready." if ok else "Fix the items above, then run this cell again.")'''

INSTALL = '''# Pinned versions of the recorded runs. torch and numpy stay as the runtime ships them: the constraints file stops pip from changing them.
from importlib.metadata import version
open("constraints.txt", "w").write("numpy==" + version("numpy") + chr(10) + "torch==" + version("torch") + chr(10))
%pip uninstall -y -q torchao
%pip install -q -c constraints.txt "transformers==5.22.1" "tokenizers==0.23.3" "huggingface_hub==2.4.0" "datasets==5.1.0" "peft==0.21.2" "accelerate==1.15.0" "lm-eval==0.4.13" "langdetect==1.0.9" "immutabledict==4.3.1" "nltk==3.10.4"
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)   # IFEval's sentence counter needs both'''

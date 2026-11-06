# Runtime check. Run this first, before the install cell. It prints the menu path that fixes each problem it finds.
import importlib.metadata as md
import os, shutil, subprocess, sys

def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"

problems = []
if sys.version_info < (3, 11):
    problems.append(f"Python {sys.version.split()[0]} is too old. transformers 5.22.1 needs 3.11 or later; 3.12 recommended.")
try:
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:
    gpus = []
print("GPUs:", gpus or "none")
if not gpus:
    problems.append("No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2. Steps 1 to 4 still run on a CPU.")
elif any("P100" in g for g in gpus):
    problems.append("P100 found. Current PyTorch wheels dropped it. Kaggle: Settings > Accelerator > GPU T4 x2.")
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
disk = shutil.disk_usage(".").free / 1e9
print(f"python {sys.version.split()[0]} | torch {version('torch')} | transformers {version('transformers')} | "
      f"peft {version('peft')} | RAM {ram:.0f} GB | free disk {disk:.0f} GB")
if ram < 12:
    problems.append("Under 12 GB of RAM. Step 7 keeps two copies of the weights on the CPU.")
for p in problems or ["Runtime OK"]:
    print(p)
